from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from grid_search import run_numeric_grid_search, supported_parameter_patterns
except Exception:
    from .grid_search import run_numeric_grid_search, supported_parameter_patterns


def _parse_paper_ids(s: str):
    if not s:
        return None
    vals = {x.strip() for x in s.split(",") if x.strip()}
    return vals or None


def main():
    ap = argparse.ArgumentParser(
        description="Offline exhaustive numeric grid search over cached Plato's Cave runs."
    )
    ap.add_argument(
        "--runs-root",
        type=str,
        default="runs/factorized_collection",
        help="Root folder containing cached factorized run outputs (summary.json, dag/, node_scores/).",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="runs/grid_search",
        help="Where grid-search outputs will be written.",
    )
    ap.add_argument(
        "--search-space",
        type=str,
        required=True,
        help="Path to JSON file defining params and constraints for the exhaustive grid.",
    )
    ap.add_argument(
        "--paper-ids",
        type=str,
        default="",
        help="Optional comma-separated paper_ids or paper_keys to restrict processing.",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Shard papers by hash for parallel work across machines.",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index (must be < num-shards).",
    )
    ap.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    ap.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on accepted configs after constraints, for smoke tests.",
    )
    ap.add_argument(
        "--reconcile",
        type=str,
        default="prefer_parents",
        choices=["prefer_parents", "prefer_children", "union"],
        help="DAGValidation reconciliation strategy.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Worker processes. Default: min(cpu_count, selected_papers).",
    )
    ap.add_argument(
        "--parallel-backend",
        type=str,
        default="auto",
        choices=["auto", "process", "thread"],
        help="Parallel backend. 'auto' prefers processes and falls back to threads if needed.",
    )
    ap.add_argument("--reuse-cache", action="store_true", help="Reuse cached per-paper grid outputs if present.")
    ap.add_argument("--force", action="store_true", help="Recompute outputs even when cached grid files exist.")
    ap.add_argument("--dry-run", action="store_true", help="Enumerate configs and write manifest/config index only.")
    ap.add_argument(
        "--list-supported-params",
        action="store_true",
        help="Print supported numeric override key patterns and exit.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_supported_params:
        print(json.dumps(supported_parameter_patterns(), indent=2, ensure_ascii=False))
        return

    if args.shard_index < 0 or args.shard_index >= max(1, args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    res = run_numeric_grid_search(
        runs_root=args.runs_root,
        out_root=args.out_root,
        search_space_path=args.search_space,
        paper_ids=_parse_paper_ids(args.paper_ids),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_papers=args.max_papers,
        max_configs=args.max_configs,
        reconcile=args.reconcile,
        max_workers=args.max_workers,
        parallel_backend=args.parallel_backend,
        reuse_cache=bool(args.reuse_cache),
        force=bool(args.force),
        verbose=bool(args.verbose),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
