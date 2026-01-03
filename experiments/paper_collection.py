"""experiments.paper_collection

Read the project's paper collection spreadsheet and normalize it to PaperRecord.

The repository currently uses an Excel workbook for systematic collection.
Batch scripts should treat this file as the source of truth for:
  - which papers to score
  - metadata (title/authors/venue/rating)

We keep this in experiments/ (not backend/) to avoid mixing UI/server code with
offline experimentation utilities.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from backend.paper_io import PaperRecord


def _norm_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def read_paper_collection_xlsx(
    xlsx_path: str,
    *,
    sheets: Optional[Iterable[str]] = None,
) -> List[PaperRecord]:
    """Read an Excel workbook and return a list of PaperRecord.

    Expected columns (case-insensitive, extra columns ignored):
      - Paper Title
      - Authors
      - URL to Paper
      - Field
      - Publication/Venue
      - Rating (good/bad/neutral, etc.)
    """

    xl = pd.ExcelFile(xlsx_path)
    sheet_names = list(xl.sheet_names)
    if sheets is not None:
        want = {s.strip() for s in sheets if s and str(s).strip()}
        sheet_names = [s for s in sheet_names if s in want]

    out: List[PaperRecord] = []
    for sheet in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if df is None or df.empty:
            continue

        # Normalize column names
        colmap = {c.lower().strip(): c for c in df.columns}
        def col(*names: str) -> Optional[str]:
            for n in names:
                k = n.lower().strip()
                if k in colmap:
                    return colmap[k]
            return None

        c_title = col("paper title", "title")
        c_auth = col("authors", "author")
        c_url = col("url to paper", "url")
        c_field = col("field")
        c_venue = col("publication/venue", "venue", "publication")
        c_rating = col("rating","overall rating","overall rating (good/neutral/bad)")

        for idx, row in df.iterrows():
            title = _norm_str(row.get(c_title)) if c_title else ""
            url = _norm_str(row.get(c_url)) if c_url else ""
            # Skip totally empty rows
            if not (title or url):
                continue
            out.append(
                PaperRecord(
                    sheet=str(sheet),
                    row_index=int(idx),
                    title=title,
                    authors=_norm_str(row.get(c_auth)) if c_auth else "",
                    url=url,
                    field=_norm_str(row.get(c_field)) if c_field else "",
                    venue=_norm_str(row.get(c_venue)) if c_venue else "",
                    rating=_norm_str(row.get(c_rating)) if c_rating else "",
                )
            )
    return out
