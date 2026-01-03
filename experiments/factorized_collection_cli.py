"""experiments.factorized_collection_cli

CLI to run factorized resampling across a paper collection.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from backend.prompts import validate_prompt_templates
from dotenv import load_dotenv

from backend.factorized_experiment import run_factorized_resampling_for_pdf
from backend.llm_client import LLMConfig, LLMClient
from backend.paper_io import guess_pdf_path

from .paper_collection import read_paper_collection_xlsx
from .plotting import save_kde_plot


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clean_floats(vals: List[float]) -> List[float]:
    out = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        out.append(fv)
    return out


def _load_env(project_root: Path, *, override: bool = False) -> None:
    """
    Load environment variables from:
      - <project_root>/.env
      - <project_root>/backend/.env
    without overriding already-exported shell env unless override=True.
    """
    load_dotenv(project_root / ".env", override=override)
    load_dotenv(project_root / "backend" / ".env", override=override)


def _setup_logger(*, log_level: str, log_file: Optional[str], console: bool = True) -> logging.Logger:
    logger = logging.getLogger("platoscave")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    # Avoid duplicate handlers if re-entered.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(fh)

    return logger


def _preflight_or_die(args: argparse.Namespace, *, logger: logging.Logger) -> None:
    """
    Fail fast with clear errors instead of generating empty run directories.
    """
    # API keys
    if args.llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Either export it in your shell or put it in <repo>/.env (or backend/.env) "
                "and run with --load-dotenv (default)."
            )
    elif args.llm_provider == "browser_use":
        if not os.getenv("BROWSER_USE_API_KEY"):
            raise RuntimeError(
                "BROWSER_USE_API_KEY is not set. "
                "Either export it in your shell or put it in <repo>/.env (or backend/.env)."
            )

    if args.retrieval_mode == "exa" and int(args.exa_k) > 0:
        if not os.getenv("EXA_API_KEY"):
            raise RuntimeError(
                "EXA_API_KEY is not set but retrieval_mode=exa and exa_k>0. "
                "Either export it or set it in <repo>/.env."
            )

    # Inputs exist?
    cx = Path(args.collection_xlsx)
    if not cx.exists():
        raise RuntimeError(f"collection-xlsx not found: {cx}")

    pr = Path(args.pdf_root)
    if not pr.exists():
        raise RuntimeError(f"pdf-root not found: {pr}")

    # PDF extraction dependency check (gives immediate actionable error)
    try:
        import fitz  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF (fitz) import failed. Install pymupdf in this env."
        ) from e

    logger.info("Preflight OK (keys present, inputs exist, fitz importable).")

    from backend.prompts import validate_prompt_templates

    errs = validate_prompt_templates()
    if errs:
        raise RuntimeError("Prompt template validation failed:\n  - " + "\n  - ".join(errs))


async def _amain(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parent.parent

    # Load .env early (unless disabled)
    if args.load_dotenv:
        _load_env(project_root, override=False)

    out_root = Path(args.out_root)
    _ensure_dir(out_root)

    log_file = args.log_file or str(out_root / "run.log")
    logger = _setup_logger(
        log_level=("DEBUG" if args.debug else args.log_level),
        log_file=log_file,
        console=(not args.no_console_log),
    )

    logger.info("Starting factorized collection run.")
    logger.info("out_root=%s", out_root)

    if args.preflight:
        _preflight_or_die(args, logger=logger)

    sheets = [s.strip() for s in args.sheets.split(",") if s.strip()] or None
    pdf_root = Path(args.pdf_root)

    records = read_paper_collection_xlsx(args.collection_xlsx, sheets=sheets)
    if not records:
        raise SystemExit("No papers found in collection")

    llm_cfg = LLMConfig(provider=args.llm_provider, model=args.model, temperature=float(args.temperature))
    llm = LLMClient(llm_cfg)

    aggregate_rows: List[Dict[str, str]] = []

    try:
        logger.info("Loaded %d paper records.", len(records))

        for idx, rec in enumerate(records, start=1):
            logger.info("[%d/%d] %s/%s %s", idx, len(records), rec.sheet, rec.paper_id, rec.title)

            pdf_path = guess_pdf_path(pdf_root, rec)
            if not pdf_path.exists():
                alt = pdf_root / pdf_path.name
                if alt.exists():
                    pdf_path = alt
                else:
                    logger.warning("Missing PDF: %s", pdf_path)
                    aggregate_rows.append(
                        {
                            "paper_id": rec.paper_id,
                            "sheet": rec.sheet,
                            "title": rec.title,
                            "rating": rec.rating,
                            "pdf_path": str(pdf_path),
                            "status": "missing_pdf",
                            "mean": "",
                            "std": "",
                            "min": "",
                            "max": "",
                            "n": "0",
                            "fail_stage": "",
                            "fail_message": "",
                        }
                    )
                    continue

            paper_out = out_root / rec.sheet / f"{rec.paper_id}__{rec.title_slug}"
            _ensure_dir(paper_out)

            # Save metadata for traceability
            (paper_out / "metadata.json").write_text(
                json.dumps(asdict(rec), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            try:
                summary = await run_factorized_resampling_for_pdf(
                    pdf_path=str(pdf_path),
                    out_dir=str(paper_out),
                    llm_cfg=llm_cfg,
                    llm=llm,  # reuse
                    paper_id=rec.paper_id,
                    retrieval_mode=str(args.retrieval_mode),
                    k_dags=int(args.k_dags),
                    m_node_resamples=int(args.m_node),
                    max_nodes=int(args.max_nodes),
                    exa_k=int(args.exa_k),
                    reconcile=str(args.reconcile),
                    node_concurrency=int(args.node_concurrency),
                    max_json_repairs=int(args.max_json_repairs),
                    reuse_cached=not args.no_cache,
                    logger=logger,
                    debug=bool(args.debug),
                )
            except Exception as e:
                tb = traceback.format_exc()
                (paper_out / "error.txt").write_text(tb, encoding="utf-8")
                logger.error("Paper failed with exception: %s: %s", type(e).__name__, e)
                logger.debug(tb)

                aggregate_rows.append(
                    {
                        "paper_id": rec.paper_id,
                        "sheet": rec.sheet,
                        "title": rec.title,
                        "rating": rec.rating,
                        "pdf_path": str(pdf_path),
                        "status": "error",
                        "mean": "",
                        "std": "",
                        "min": "",
                        "max": "",
                        "n": "0",
                        "fail_stage": "paper_exception",
                        "fail_message": f"{type(e).__name__}: {e}",
                    }
                )
                if args.fail_fast:
                    raise
                continue

            g = summary.get("global", {})
            n = int(g.get("n", 0) or 0)

            diag = summary.get("diagnostics", {}) or {}
            first_fail = diag.get("first_failure", {}) or {}
            fail_stage = str(first_fail.get("stage") or "")
            fail_msg = str(first_fail.get("message") or "")

            if n == 0:
                logger.warning(
                    "No successful trials (paper_id=%s). first_failure=%s | %s",
                    rec.paper_id,
                    fail_stage,
                    fail_msg[:300],
                )
                if args.fail_fast:
                    raise RuntimeError(f"Fail-fast: {rec.paper_id} produced no successful trials.")

            aggregate_rows.append(
                {
                    "paper_id": rec.paper_id,
                    "sheet": rec.sheet,
                    "title": rec.title,
                    "rating": rec.rating,
                    "pdf_path": str(pdf_path),
                    "status": "ok" if n else "no_successful_trials",
                    "mean": "" if math.isnan(float(g.get("mean", float("nan")))) else f"{float(g.get('mean')):.8f}",
                    "std": "" if math.isnan(float(g.get("std", float("nan")))) else f"{float(g.get('std')):.8f}",
                    "min": "" if math.isnan(float(g.get("min", float("nan")))) else f"{float(g.get('min')):.8f}",
                    "max": "" if math.isnan(float(g.get("max", float("nan")))) else f"{float(g.get('max')):.8f}",
                    "n": str(n),
                    "fail_stage": fail_stage,
                    "fail_message": fail_msg[:500],
                }
            )

            # KDE plot for this paper (successful scores only)
            csv_path = paper_out / "graph_scores.csv"
            if csv_path.exists():
                scores: List[float] = []
                with csv_path.open("r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        if row.get("success") != "1":
                            continue
                        if row.get("graph_score"):
                            scores.append(float(row["graph_score"]))
                scores = _clean_floats(scores)
                save_kde_plot(
                    values=scores,
                    out_path=str(paper_out / "kde_graph_score.png"),
                    title=f"KDE graph_score: {rec.paper_id}",
                )

        # Write aggregate CSV
        agg_csv = out_root / "papers_summary.csv"
        fieldnames = [
            "paper_id",
            "sheet",
            "title",
            "rating",
            "pdf_path",
            "status",
            "mean",
            "std",
            "min",
            "max",
            "n",
            "fail_stage",
            "fail_message",
        ]
        with agg_csv.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for r in aggregate_rows:
                wr.writerow(r)

        (out_root / "run_config.json").write_text(
            json.dumps(
                {
                    "collection_xlsx": args.collection_xlsx,
                    "sheets": sheets,
                    "pdf_root": str(pdf_root),
                    "k_dags": int(args.k_dags),
                    "m_node": int(args.m_node),
                    "max_nodes": int(args.max_nodes),
                    "exa_k": int(args.exa_k),
                    "retrieval_mode": str(args.retrieval_mode),
                    "node_concurrency": int(args.node_concurrency),
                    "max_json_repairs": int(args.max_json_repairs),
                    "reconcile": str(args.reconcile),
                    "llm": {"provider": args.llm_provider, "model": args.model, "temperature": float(args.temperature)},
                    "logging": {"log_level": ("DEBUG" if args.debug else args.log_level), "log_file": log_file},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info("Wrote: %s", agg_csv)
        return 0

    finally:
        await llm.aclose()


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--collection-xlsx", required=True)
    ap.add_argument("--sheets", default="", help="Comma-separated list; default = all sheets")
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--k-dags", type=int, default=3)
    ap.add_argument("--m-node", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=10)
    ap.add_argument("--exa-k", type=int, default=6)
    ap.add_argument("--retrieval-mode", type=str, default="llm", choices=["exa", "llm"])
    ap.add_argument("--node-concurrency", type=int, default=3)
    ap.add_argument(
        "--max-json-repairs",
        type=int,
        default=int(os.getenv("MAX_JSON_REPAIRS", "1")),
        help="Iterative LLM repair attempts for malformed JSON outputs",
    )

    ap.add_argument("--reconcile", type=str, default="prefer_parents")
    ap.add_argument("--no-cache", action="store_true")

    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "browser_use"])

    # Debug / logging / safety
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", type=str, default="", help="Default: <out-root>/run.log")
    ap.add_argument("--debug", action="store_true", help="Convenience flag: sets log level to DEBUG and writes tracebacks")
    ap.add_argument("--no-console-log", action="store_true", help="Disable console logging (still logs to file)")

    ap.add_argument("--load-dotenv", action="store_true", default=True)
    ap.add_argument("--no-load-dotenv", action="store_false", dest="load_dotenv")
    ap.add_argument("--preflight", action="store_true", default=True)
    ap.add_argument("--no-preflight", action="store_false", dest="preflight")
    ap.add_argument("--fail-fast", action="store_true", help="Stop immediately on first paper error or n=0 result.")

    args = ap.parse_args()

    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl-C). Partial outputs remain under --out-root.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
