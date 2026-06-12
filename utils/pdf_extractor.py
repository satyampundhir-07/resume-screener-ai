"""
utils/pdf_extractor.py
-----------------------
Extracts plain text from PDF and DOCX files.
Uses PyMuPDF (fitz) for PDFs — fast, dependency-light, no Java needed.

Author: Resume Screener ML Pipeline
"""

import io
import re

# ── PDF extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF byte stream.

    Parameters
    ----------
    file_bytes : bytes  Raw bytes of the uploaded PDF.

    Returns
    -------
    str  Concatenated text of every page, separated by newlines.
    """
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return _clean_extracted(text)
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract all text from a DOCX byte stream.

    Parameters
    ----------
    file_bytes : bytes  Raw bytes of the uploaded DOCX.

    Returns
    -------
    str  Concatenated paragraph text.
    """
    try:
        from docx import Document
        doc  = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return _clean_extracted(text)
    except Exception as exc:
        return f"[DOCX extraction error: {exc}]"


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Auto-detect file type and extract text.

    Parameters
    ----------
    file_bytes : bytes
    filename   : str   Original filename (used for extension detection).

    Returns
    -------
    str  Extracted text.
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        # Try PDF first, then plain text fallback
        try:
            return extract_text_from_pdf(file_bytes)
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _clean_extracted(text: str) -> str:
    """Collapse excessive blank lines and strip leading/trailing space."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
