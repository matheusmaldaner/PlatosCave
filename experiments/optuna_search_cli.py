from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from optuna_search import run_optuna_search, supported_parameter_patterns
except Exception:
    from .optuna_search import run_optuna_search, supported_parameter_patterns


def _parse_paper_ids(s: str):
    if not s:
        return None
    vals = {x.strip() for x in s.split(",") if x.strip()}
    return vals or None


def main():
    ap = argparse.ArgumentParser(
        description="Optuna-based offline hyperparameter search over cached Plato's Cave runs."
    )
    ap.add_argument("--runs-root", type=str, default="runs/factorized_collection")
    ap.add_argument("--out-root", type=str, default="runs/optuna_search")
    ap.add_argument("--search-space", type=str, required=True)
    ap.add_argument("--paper-ids", type=str, default="")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument(
        "--reconcile",
        type=str,
        default="prefer_parents",
        choices=["prefer_parents", "prefer_children", "union"],
    )
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel Optuna trial workers (threads in-process).",
    )
    ap.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optional Optuna storage URL. Default: sqlite file under the output directory.",
    )
    ap.add_argument(
        "--study-name",
        type=str,
        default="",
        help="Optional explicit Optuna study name. Default: search-space name.",
    )
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-supported-params", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_supported_params:
        print(json.dumps(supported_parameter_patterns(), indent=2, ensure_ascii=False))
        return

    if args.shard_index < 0 or args.shard_index >= max(1, args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    res = run_optuna_search(
        runs_root=args.runs_root,
        out_root=args.out_root,
        search_space_path=args.search_space,
        paper_ids=_parse_paper_ids(args.paper_ids),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_papers=args.max_papers,
        reconcile=args.reconcile,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=(args.storage or None),
        study_name=(args.study_name or None),
        reuse_cache=bool(args.reuse_cache),
        force=bool(args.force),
        verbose=bool(args.verbose),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
