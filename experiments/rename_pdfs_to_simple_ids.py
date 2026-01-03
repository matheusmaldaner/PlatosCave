"""experiments.rename_pdfs_to_simple_ids

One-time utility to rename legacy hashed PDF filenames to the new human-simple
naming convention.

Legacy:
  <pdf_root>/<sheet>/<idx>_<hash>__<title_slug>.pdf
New:
  <pdf_root>/<sheet>/<idx>__<title_slug>.pdf

The scoring code can already locate legacy files without renaming. This script
exists purely to make the on-disk PDF collection easier to read and maintain.

Usage example:
  python -m experiments.rename_pdfs_to_simple_ids \
    --collection-xlsx "Paper collection.xlsx" \
    --pdf-root "data/pdfs" \
    --apply

If --apply is omitted, the script runs in dry-run mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.paper_io import safe_slug
from .paper_collection import read_paper_collection_xlsx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-xlsx", required=True)
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--sheets", default="")
    ap.add_argument("--apply", action="store_true", help="Perform renames (default: dry-run)")
    args = ap.parse_args()

    sheets = [s.strip() for s in args.sheets.split(",") if s.strip()] or None
    pdf_root = Path(args.pdf_root)

    records = read_paper_collection_xlsx(args.collection_xlsx, sheets=sheets)
    if not records:
        raise SystemExit("No papers found in collection")

    n_renamed = 0
    n_skipped = 0

    for rec in records:
        sheet_dir = pdf_root / rec.sheet
        if not sheet_dir.exists():
            continue

        # New preferred path
        title_slug = safe_slug(rec.title or f"paper-{rec.row_index:03d}")
        new_path = sheet_dir / f"{rec.row_index:03d}__{title_slug}.pdf"
        if new_path.exists():
            n_skipped += 1
            continue

        # Legacy matches
        legacy_matches = sorted(sheet_dir.glob(f"{rec.row_index:03d}_*__*.pdf"))
        if not legacy_matches:
            continue

        # Prefer a match containing the slug, else take the first.
        legacy = None
        for m in legacy_matches:
            if title_slug and title_slug in m.name:
                legacy = m
                break
        legacy = legacy or legacy_matches[0]

        if args.apply:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(new_path)
            print(f"[RENAMED] {legacy.name} -> {new_path.name}")
        else:
            print(f"[DRYRUN] {legacy.name} -> {new_path.name}")
        n_renamed += 1

    print(f"Done. candidates={n_renamed} skipped(already_simple)={n_skipped} apply={bool(args.apply)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
