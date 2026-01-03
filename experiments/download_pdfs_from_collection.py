"""experiments.download_pdfs_from_collection

Mass-download PDFs for the paper collection spreadsheet into a stable local
folder structure.

This is intentionally decoupled from scoring so you can:
  - run downloads once
  - manually fix any paywalled / broken links
  - rerun scoring without hitting the network

Output convention:
  <pdf_root>/<sheet>/<paper_id>__<title_slug>.pdf

Notes
 - We only attempt lightweight heuristics to get direct PDFs.
 - If a URL is not directly downloadable, we skip it and log the reason.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import requests

from backend.paper_io import find_existing_pdf, guess_pdf_path
from .paper_collection import read_paper_collection_xlsx


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_pdf_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    # arXiv
    m = re.match(r"^https?://arxiv\.org/abs/(?P<id>[0-9\.]+)(v\d+)?$", u)
    if m:
        return f"https://arxiv.org/pdf/{m.group('id')}.pdf"
    if "arxiv.org/pdf/" in u and not u.endswith(".pdf"):
        return u + ".pdf"

    # OpenReview (common patterns)
    if "openreview.net" in u and "pdf" not in u:
        # e.g., https://openreview.net/forum?id=XYZ -> https://openreview.net/pdf?id=XYZ
        m2 = re.search(r"[?&]id=([^&]+)", u)
        if m2:
            return f"https://openreview.net/pdf?id={m2.group(1)}"

    return u


def _is_probably_pdf(resp: requests.Response) -> bool:
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/pdf" in ct:
        return True
    if resp.url.lower().endswith(".pdf"):
        return True
    return False


def download_one(url: str, out_path: Path, *, timeout_s: int = 60) -> str:
    """Return status string."""
    u = _normalize_pdf_url(url)
    if not u:
        return "missing_url"

    try:
        # HEAD first
        h = requests.head(u, allow_redirects=True, timeout=timeout_s)
        if h.status_code >= 400:
            # Some sites block HEAD; fall back to GET.
            h = None
        elif h is not None and not _is_probably_pdf(h):
            # Might still be a PDF behind redirect; proceed with GET.
            pass

        r = requests.get(u, allow_redirects=True, timeout=timeout_s)
        if r.status_code >= 400:
            return f"http_{r.status_code}"
        if not _is_probably_pdf(r):
            return "not_pdf"

        _ensure_dir(out_path.parent)
        out_path.write_bytes(r.content)
        return "downloaded"
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        return f"error:{type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-xlsx", required=True)
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--sheets", default="")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    sheets = [s.strip() for s in args.sheets.split(",") if s.strip()] or None
    pdf_root = Path(args.pdf_root)
    _ensure_dir(pdf_root)

    records = read_paper_collection_xlsx(args.collection_xlsx, sheets=sheets)
    if not records:
        raise SystemExit("No papers found in collection")

    ok = 0
    for rec in records:
        # Preferred output path (may differ from an existing legacy path).
        out_path = pdf_root / rec.sheet / f"{rec.paper_id}__{rec.title_slug}.pdf"

        if args.skip_existing:
            existing = find_existing_pdf(pdf_root, rec)
            if existing is not None:
                print(f"[SKIP exists] {rec.sheet} | {rec.paper_id} | {existing.name}")
                continue
        status = download_one(rec.url, out_path, timeout_s=int(args.timeout))
        print(f"[{status}] {rec.sheet} | {rec.paper_id} | {out_path.name}")
        if status == "downloaded":
            ok += 1

    print(f"Downloaded {ok} PDFs into: {pdf_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
