
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ablation_studies import run_rescoring_study, list_studies, AblationError
except Exception:
    from .ablation_studies import run_rescoring_study, list_studies, AblationError


def _parse_study_ids(s: str):
    ids = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        ids.append(int(tok))
    return ids


def _parse_paper_ids(s: str):
    if not s:
        return None
    vals = {x.strip() for x in s.split(",") if x.strip()}
    return vals or None


def main():
    ap = argparse.ArgumentParser(
        description="Offline ablation studies for Plato's Cave factorized KxM cached runs."
    )
    ap.add_argument("--runs-root", type=str, default="runs/factorized_collection",
                    help="Root folder containing cached factorized run outputs (summary.json, dag/, node_scores/).")
    ap.add_argument("--out-root", type=str, default="runs/ablation_studies",
                    help="Where ablation outputs will be written.")
    ap.add_argument("--studyIDs", type=str, default="1,2,3,4,5,6",
                    help="Comma-separated study IDs to run, e.g. --studyIDs 1,3,6")
    ap.add_argument("--list-studies", action="store_true", help="Print study registry JSON and exit.")
    ap.add_argument("--paper-ids", type=str, default="",
                    help="Optional comma-separated paper_ids or paper_keys to restrict processing.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Shard papers by hash for parallel work across students/machines.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="0-based shard index (must be < num-shards).")
    ap.add_argument("--max-papers", type=int, default=None,
                    help="Optional cap for quick smoke tests.")
    ap.add_argument("--reconcile", type=str, default="prefer_parents",
                    choices=["prefer_parents", "prefer_children", "union"],
                    help="DAGValidation reconciliation strategy for rescoring studies.")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="Reuse per-study/per-paper outputs if present.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute outputs even when cached ablation files exist.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_studies:
        print(json.dumps(list_studies(), indent=2, ensure_ascii=False))
        return

    if args.shard_index < 0 or args.shard_index >= max(1, args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")

    study_ids = _parse_study_ids(args.studyIDs)
    if not study_ids:
        raise SystemExit("No study IDs provided.")
    paper_ids = _parse_paper_ids(args.paper_ids)

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    results = []
    for sid in study_ids:
        if args.verbose:
            print(f"Running study {sid} ...")
        try:
            res = run_rescoring_study(
                runs_root=args.runs_root,
                out_root=args.out_root,
                study_id=sid,
                paper_ids=paper_ids,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
                max_papers=args.max_papers,
                reconcile=args.reconcile,
                reuse_cache=bool(args.reuse_cache),
                force=bool(args.force),
                verbose=bool(args.verbose),
            )
            results.append({"study_id": sid, "ok": True, **res})
        except AblationError as e:
            results.append({"study_id": sid, "ok": False, "error": str(e)})
        except Exception as e:
            results.append({"study_id": sid, "ok": False, "error": f"{type(e).__name__}: {e}"})

    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
