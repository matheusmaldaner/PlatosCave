"""backend.paper_io

I/O helpers for batch experiments.

This module intentionally duplicates a small amount of logic from backend/main.py
so that batch experiments can:
  - avoid importing backend/main.py (which streams UI protocol JSON to stdout)
  - avoid pulling in Browser/Agent dependencies when they are not needed

Primary responsibilities:
  - Extract text from a local PDF (PyMuPDF)
  - Provide stable IDs / safe filenames for papers
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def safe_slug(text: str, *, max_len: int = 80) -> str:
    """Convert arbitrary text to a filesystem-friendly slug."""
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\- _]", "", t)
    t = t.replace(" ", "-")
    t = re.sub(r"-+", "-", t).strip("-")
    return t[:max_len] if len(t) > max_len else t


def stable_id(*parts: str, n: int = 10) -> str:
    """Stable short ID for filenames (sha1 over the joined parts)."""
    h = hashlib.sha1("|".join([p or "" for p in parts]).encode("utf-8"), usedforsecurity=False)
    return h.hexdigest()[:n]


@dataclass(frozen=True)
class PaperRecord:
    sheet: str
    row_index: int
    title: str
    authors: str
    url: str
    field: str
    venue: str
    rating: str

    @property
    def paper_id(self) -> str:
        """Human-simple paper identifier.

        We use the spreadsheet row index (zero-based) padded to 3 digits.

        Rationale:
          - Easy for humans to read and manually manage files.
          - Unambiguous within a sheet because PDFs are stored under
            <pdf_root>/<sheet>/.

        Important:
          - If you reorder rows in the spreadsheet, IDs will change.
            Treat the spreadsheet as the canonical ordering.
        """
        return f"{self.row_index:03d}"

    @property
    def title_slug(self) -> str:
        return safe_slug(self.title or f"paper-{self.paper_id}")


def guess_pdf_path(pdf_root: Path, rec: PaperRecord) -> Path:
    """Default convention used by experiments/download_pdfs_from_collection.py.

    Primary convention (human-simple):
      <pdf_root>/<sheet>/<row_index:03d>__<title_slug>.pdf

    Backward compatibility:
      If a legacy hashed filename exists (e.g., 003_ab12cd34ef__title.pdf), we
      return that path so existing downloads continue to work without renaming.
    """
    sheet_dir = pdf_root / rec.sheet
    preferred = sheet_dir / f"{rec.paper_id}__{rec.title_slug}.pdf"
    if preferred.exists():
        return preferred

    # Human-friendly fallbacks:
    #   - <idx>__anything.pdf
    #   - <idx>.pdf
    if sheet_dir.exists():
        simple_matches = sorted(sheet_dir.glob(f"{rec.row_index:03d}__*.pdf"))
        if len(simple_matches) == 1:
            return simple_matches[0]
        if len(simple_matches) > 1:
            for m in simple_matches:
                if rec.title_slug and rec.title_slug in m.name:
                    return m
            return simple_matches[0]

        plain = sheet_dir / f"{rec.row_index:03d}.pdf"
        if plain.exists():
            return plain

    # Legacy pattern: <idx>_<hash>__<slug>.pdf
    if sheet_dir.exists():
        legacy_matches = sorted(sheet_dir.glob(f"{rec.row_index:03d}_*__*.pdf"))
        if len(legacy_matches) == 1:
            return legacy_matches[0]
        if len(legacy_matches) > 1:
            # Prefer a match that includes the slug (when possible).
            for m in legacy_matches:
                if rec.title_slug and rec.title_slug in m.name:
                    return m
            return legacy_matches[0]

    return preferred


def find_existing_pdf(pdf_root: Path, rec: PaperRecord) -> Optional[Path]:
    """Return an existing PDF path for this record, if any.

    This is useful for download scripts that want to honor --skip-existing
    without redownloading when a legacy filename exists.
    """
    p = guess_pdf_path(pdf_root, rec)
    return p if p.exists() else None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF (fitz)."""
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF (pymupdf) is required to extract PDF text") from e

    doc = fitz.open(str(p))
    try:
        text_content = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text() or ""
            if page_text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
        return "\n\n".join(text_content)
    finally:
        doc.close()
