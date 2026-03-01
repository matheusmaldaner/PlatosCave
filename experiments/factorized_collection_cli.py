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
from dotenv import load_dotenv
from backend.prompts import validate_prompt_templates
from backend.factorized_experiment import run_factorized_resampling_for_pdf
from backend.llm_client import LLMConfig, LLMClient, set_global_llm_concurrency
from backend.paper_io import guess_pdf_path
from backend._perf import PerfWriter, new_run_id, timed
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    level = getattr(logging, log_level.upper(), logging.INFO)

    # We keep "platoscave" as the user-facing logger, but we also configure root so
    # that logs from other modules (e.g., backend.llm_client) appear in the same outputs.
    logger = logging.getLogger("platoscave")
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicates for platoscave messages

    # Avoid duplicate handlers if re-entered.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Include logger name so you can see backend.llm_client, backend.factorized_experiment, etc.
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handlers: list[logging.Handler] = []

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        ch.setLevel(level)
        handlers.append(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        handlers.append(fh)

    for h in handlers:
        logger.addHandler(h)

    # ---- Key fix: configure ROOT logger with the same handlers ----
    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing root handlers to avoid duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)

    for h in handlers:
        root.addHandler(h)

    # Optional: reduce noise from chatty libraries (tune as desired)
    logging.getLogger("asyncio").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("openai").setLevel(max(level, logging.WARNING))
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))

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

    timeout_s = float(args.llm_timeout_s) if float(args.llm_timeout_s) > 0 else None

    # Model routing:
    # - New: optionally override with --model-dag (DAG extraction) and/or --model-node (node scoring).
    model_dag = (getattr(args, "model_dag", "") or "").strip() or None
    model_node = (getattr(args, "model_node", "") or "").strip() or None
    if model_dag is None and model_node is None:
        model_dag = str(args.model)
        model_node = str(args.model)
    else:
        model_dag = model_dag or str(args.model)
        model_node = model_node or str(args.model)

    common_cfg_kwargs = dict(
        provider=args.llm_provider,
        temperature=float(args.temperature),
        timeout_s=timeout_s,
        max_retries=int(args.llm_max_retries),
        min_backoff_s=float(args.llm_min_backoff_s),
        max_backoff_s=float(args.llm_max_backoff_s),
        default_rate_limit_cooldown_s=float(args.llm_rate_limit_cooldown_s),
    )

    llm_cfg_dag = LLMConfig(model=str(model_dag), **common_cfg_kwargs)
    llm_cfg_node = LLMConfig(model=str(model_node), **common_cfg_kwargs)

    llm_dag = LLMClient(llm_cfg_dag)
    llm_node = LLMClient(llm_cfg_node)


    run_id = new_run_id()
    perf = PerfWriter(path=str(out_root / "perf.jsonl"), enabled=bool(args.perf_jsonl))

    # Global cap on OpenAI calls across the whole run (paper-level + node-level combined)
    set_global_llm_concurrency(int(args.llm_concurrency))

    logger.info(
        "run.meta | run_id=%s paper_concurrency=%s node_concurrency=%s llm_concurrency=%s k_dags=%s m_node=%s max_nodes=%s reuse_cached=%s model_dag=%s model_node=%s",
        run_id,
        int(args.paper_concurrency),
        int(args.node_concurrency),
        int(args.llm_concurrency),
        int(args.k_dags),
        int(args.m_node),
        int(args.max_nodes),
        (not args.no_cache),
        str(model_dag),
        str(model_node),
    )

    aggregate_rows: List[Dict[str, str]] = []

    try:
        logger.info("Loaded %d paper records.", len(records))

        paper_sem = asyncio.Semaphore(max(1, int(args.paper_concurrency)))

        async def _process_one(idx: int, rec) -> Dict[str, str]:
            # Limit how many papers are in-flight at once
            async with paper_sem:
                ctx = {
                    "run_id": run_id,
                    "paper_i": idx,
                    "paper_n": len(records),
                    "paper_id": rec.paper_id,
                    "sheet": rec.sheet,
                }

                with timed(logger, "paper.total", warn_ms=300_000, perf=perf, **ctx):
                    logger.info("[%d/%d] %s/%s %s", idx, len(records), rec.sheet, rec.paper_id, rec.title)

                    try:
                        pdf_path = guess_pdf_path(pdf_root, rec)
                        if not pdf_path.exists():
                            alt = pdf_root / pdf_path.name
                            if alt.exists():
                                pdf_path = alt
                            else:
                                logger.warning("Missing PDF: %s", pdf_path)
                                return {
                                    "paper_id": rec.paper_id,
                                    "sheet": rec.sheet,
                                    "title": rec.title,
                                    "rating": rec.rating,
                                    "pdf_path": str(pdf_path),
                                    "status": "missing_pdf",
                                    "mean": "",
                                    "stdev": "",
                                    "n": "0",
                                    "min": "",
                                    "max": "",
                                    "first_failure_stage": "missing_pdf",
                                    "first_failure_message": "",
                                }

                        # Output folder should be stable, filesystem-safe, and consistent with backend.paper_io conventions.
                        paper_out = out_root / rec.sheet / f"{rec.paper_id}__{rec.title_slug}"
                        _ensure_dir(paper_out)

                        # Write a small record.json once
                        try:
                            (paper_out / "record.json").write_text(
                                json.dumps(asdict(rec), indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass  # non-fatal

                        with timed(logger, "paper.run_factorized", warn_ms=300_000, perf=perf, **ctx):
                            summary = await run_factorized_resampling_for_pdf(
                                pdf_path=str(pdf_path),
                                out_dir=str(paper_out),
                                llm_cfg=llm_cfg_node,
                                llm_cfg_dag=llm_cfg_dag,
                                llm_dag=llm_dag,
                                llm_node=llm_node,
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
                                perf=perf,
                                run_id=run_id,
                                paper_index=idx,
                            )

                        diag = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}

                        # Backward-compatible stats resolution:
                        # - New schema: top-level "global" (preferred)
                        # - Old schema: "stats"
                        if isinstance(summary, dict):
                            stats = summary.get("global")
                            if not isinstance(stats, dict):
                                stats = summary.get("stats")
                            if not isinstance(stats, dict):
                                stats = {}
                        else:
                            stats = {}

                        def _fmt_num(x) -> str:
                            import math
                            if x is None:
                                return ""
                            try:
                                xf = float(x)
                                if math.isnan(xf) or math.isinf(xf):
                                    return ""
                                return str(xf)
                            except Exception:
                                return str(x)

                        n = str(int(stats.get("n") or 0))
                        mean = _fmt_num(stats.get("mean"))
                        # backend uses "std"
                        stdev = _fmt_num(stats.get("std") if "std" in stats else stats.get("stdev"))
                        vmin = _fmt_num(stats.get("min"))
                        vmax = _fmt_num(stats.get("max"))

                        num_trials = diag.get("num_trials", "")
                        num_success = diag.get("num_success", "")
                        failure_counts = diag.get("failure_counts", {})
                        first_fail = diag.get("first_failure", {}) or {}
                        fail_stage = str(first_fail.get("stage") or "")
                        fail_msg = str(first_fail.get("message") or "")

                        logger.info(
                            "paper.result | paper_id=%s trials=%s success=%s n=%s mean=%s std=%s first_failure=%s",
                            rec.paper_id, num_trials, num_success, n, mean, stdev, fail_stage
                        )

                        if str(num_success) in ("0", "", "None"):
                            logger.warning(
                                "paper.no_success | paper_id=%s failure_counts=%s first_failure=%s | %s",
                                rec.paper_id, dict(failure_counts) if isinstance(failure_counts, dict) else failure_counts,
                                fail_stage, fail_msg[:300]
                            )

                        first_fail = diag.get("first_failure", {}) or {}
                        fail_stage = str(first_fail.get("stage") or "")
                        fail_msg = str(first_fail.get("message") or "")

                        if n == "0":
                            logger.warning(
                                "No successful trials (paper_id=%s). first_failure=%s | %s",
                                rec.paper_id,
                                fail_stage,
                                fail_msg[:300],
                            )
                            if args.fail_fast:
                                raise RuntimeError(f"Fail-fast: {rec.paper_id} produced no successful trials.")
                        
                        status = "ok" if n != "0" else "no_success"

                        return {
                            "paper_id": rec.paper_id,
                            "sheet": rec.sheet,
                            "title": rec.title,
                            "rating": rec.rating,
                            "pdf_path": str(pdf_path),
                            "status": status,
                            "mean": mean,
                            "stdev": stdev,
                            "n": n,
                            "min": vmin,
                            "max": vmax,
                            "first_failure_stage": fail_stage,
                            "first_failure_message": fail_msg,
                        }

                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error("Paper failed with exception: %s: %s", type(e).__name__, e)
                        logger.debug(tb)
                        return {
                            "paper_id": rec.paper_id,
                            "sheet": rec.sheet,
                            "title": rec.title,
                            "rating": rec.rating,
                            "pdf_path": "",
                            "status": "error",
                            "mean": "",
                            "stdev": "",
                            "n": "0",
                            "min": "",
                            "max": "",
                            "first_failure_stage": "exception",
                            "first_failure_message": f"{type(e).__name__}: {e}",
                        }

        # Fan out papers concurrently (bounded by paper_sem)
        with timed(logger, "collection.process_all", warn_ms=3_600_000, perf=perf, run_id=run_id, n_papers=len(records)):
            tasks = [asyncio.create_task(_process_one(i, rec)) for i, rec in enumerate(records, start=1)]
            results = await asyncio.gather(*tasks)
            aggregate_rows.extend(results)

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
            "stdev",
            "min",
            "max",
            "n",
            "first_failure_stage",
            "first_failure_message",
        ]
        with agg_csv.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            wr.writeheader()
            for r in aggregate_rows:
                wr.writerow(r)

        logger.info("Wrote: %s", agg_csv)
        return 0

    finally:
        await llm_dag.aclose()
        if llm_node is not llm_dag:
            await llm_node.aclose()


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
    ap.add_argument("--max-json-repairs", type=int, default=int(os.getenv("MAX_JSON_REPAIRS", "1")))

    ap.add_argument("--reconcile", type=str, default="prefer_parents")
    ap.add_argument("--no-cache", action="store_true")

    ap.add_argument("--model", type=str, default="gpt-5-nano")
    ap.add_argument( "--model-dag", type=str, default="", help="Optional override model for DAG extraction only (leave blank to use --model)")
    ap.add_argument("--model-node",type=str,default="",help="Optional override model for node scoring only (leave blank to use --model)")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "browser_use"])

    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-file", type=str, default="", help="Default: <out-root>/run.log")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-console-log", action="store_true")

    ap.add_argument("--load-dotenv", action="store_true", default=True)
    ap.add_argument("--no-load-dotenv", action="store_false", dest="load_dotenv")
    ap.add_argument("--preflight", action="store_true", default=True)
    ap.add_argument("--no-preflight", action="store_false", dest="preflight")
    ap.add_argument("--fail-fast", action="store_true")

    ap.add_argument("--paper-concurrency", type=int, default=1)
    ap.add_argument("--llm-concurrency", type=int, default=8)
    ap.add_argument("--perf-jsonl", action="store_true")

    ap.add_argument("--llm-timeout-s", type=float, default=0.0, help="Per-call timeout (0 = no timeout)")
    ap.add_argument("--llm-max-retries", type=int, default=8, help="Retries per LLM call on rate limits/transients")
    ap.add_argument("--llm-min-backoff-s", type=float, default=1.0, help="Min backoff for retries")
    ap.add_argument("--llm-max-backoff-s", type=float, default=60.0, help="Max backoff for retries")
    ap.add_argument("--llm-rate-limit-cooldown-s", type=float, default=10.0, help="Cooldown used when rate-limited without Retry-After")


    args = ap.parse_args()

    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl-C). Partial outputs remain under --out-root.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
