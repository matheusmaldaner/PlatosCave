"""experiments.factorized_collection_cli

CLI to run factorized resampling across a paper collection.

Usage (example):
  python -m experiments.factorized_collection_cli \
    --collection-xlsx "Paper collection.xlsx" \
    --pdf-root "data/pdfs" \
    --out-root "runs/2025-12-28" \
    --k-dags 5 \
    --m-node 3 \
    --max-nodes 10

This script:
  - reads Paper collection.xlsx
  - locates each paper's local PDF (downloaded separately)
  - runs K DAG resamples + M node-score resamples per DAG
  - writes per-paper outputs and an aggregate CSV summary
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from backend.factorized_experiment import run_factorized_resampling_for_pdf
from backend.llm_client import LLMConfig, LLMClient
from backend.paper_io import PaperRecord, guess_pdf_path

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

async def _amain(args: argparse.Namespace) -> int:
    sheets = [s.strip() for s in args.sheets.split(",") if s.strip()] or None
    pdf_root = Path(args.pdf_root)
    out_root = Path(args.out_root)
    _ensure_dir(out_root)

    records = read_paper_collection_xlsx(args.collection_xlsx, sheets=sheets)
    if not records:
        raise SystemExit("No papers found in collection")

    llm_cfg = LLMConfig(provider=args.llm_provider, model=args.model, temperature=float(args.temperature))
    llm = LLMClient(llm_cfg)

    aggregate_rows: List[Dict[str, str]] = []

    try:
        for rec in records:
            pdf_path = guess_pdf_path(pdf_root, rec)
            if not pdf_path.exists():
                # Also allow user to place PDFs directly under pdf_root with the same filename.
                alt = pdf_root / pdf_path.name
                if alt.exists():
                    pdf_path = alt
                else:
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

            # IMPORTANT: await (do NOT asyncio.run per paper)
            try:
                summary = await run_factorized_resampling_for_pdf(
                    pdf_path=str(pdf_path),
                    out_dir=str(paper_out),
                    llm_cfg=llm_cfg,
                    llm=llm,  # reuse the one client you already created
                    paper_id=rec.paper_id,  # ensures traceability and avoids undefined paper_id issues
                    retrieval_mode=str(args.retrieval_mode),
                    k_dags=int(args.k_dags),
                    m_node_resamples=int(args.m_node),
                    max_nodes=int(args.max_nodes),
                    exa_k=int(args.exa_k),
                    reconcile=str(args.reconcile),
                    node_concurrency=int(args.node_concurrency),
                    max_json_repairs=int(args.max_json_repairs),
                    reuse_cached=not args.no_cache,
                )

            except Exception as e:
                # Record and continue to next paper
                (paper_out / "error.txt").write_text(
                    f"{type(e).__name__}: {e}\n",
                    encoding="utf-8",
                )
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
                    }
                )
                continue

            g = summary.get("global", {})
            aggregate_rows.append(
                {
                    "paper_id": rec.paper_id,
                    "sheet": rec.sheet,
                    "title": rec.title,
                    "rating": rec.rating,
                    "pdf_path": str(pdf_path),
                    "status": "ok" if g.get("n", 0) else "no_successful_trials",
                    "mean": "" if math.isnan(float(g.get("mean", float("nan")))) else f"{float(g.get('mean')):.8f}",
                    "std": "" if math.isnan(float(g.get("std", float("nan")))) else f"{float(g.get('std')):.8f}",
                    "min": "" if math.isnan(float(g.get("min", float("nan")))) else f"{float(g.get('min')):.8f}",
                    "max": "" if math.isnan(float(g.get("max", float("nan")))) else f"{float(g.get('max')):.8f}",
                    "n": str(int(g.get("n", 0))),
                }
            )

            # KDE plot for this paper
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
                    "node_concurrency": int(args.node_concurrency),
                    "max_json_repairs": int(args.max_json_repairs),
                    "reconcile": str(args.reconcile),
                    "llm": {"model": args.model, "temperature": float(args.temperature)},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"Wrote: {agg_csv}")
        return 0

    finally:
        # Ensure underlying async HTTP clients close while event loop is alive
        await llm.aclose()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-xlsx", required=True)
    ap.add_argument("--sheets", default="", help="Comma-separated list; default = all sheets")
    ap.add_argument("--pdf-root", required=True, help="Root directory containing PDFs")
    ap.add_argument("--out-root", required=True, help="Root directory for experiment outputs")

    ap.add_argument("--k-dags", type=int, default=3)
    ap.add_argument("--m-node", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=10)
    ap.add_argument("--exa-k", type=int, default=6)
    ap.add_argument("--retrieval-mode", type=str, default="llm", choices=["exa", "llm"])
    ap.add_argument("--node-concurrency", type=int, default=3)
    ap.add_argument( "--max-json-repairs", type=int, default=int(os.getenv("MAX_JSON_REPAIRS", "1")), help="Number of iterative LLM repair attempts for malformed JSON outputs")

    ap.add_argument("--reconcile", type=str, default="prefer_parents")
    ap.add_argument("--no-cache", action="store_true", help="Disable reuse of cached artifacts")

    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "browser_use"])


    args = ap.parse_args()

    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl-C). Partial outputs remain under --out-root.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
