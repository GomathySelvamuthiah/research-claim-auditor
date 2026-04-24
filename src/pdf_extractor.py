"""
pdf_extractor.py

Extracts plain text from academic paper PDFs and isolates the introduction
section for downstream claim extraction.
"""

import re


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF byte string using pypdf."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


_INTRO_HEADERS = re.compile(
    r"^\s*(1\.?\s+)?introduction\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_NEXT_SECTION_HEADERS = re.compile(
    r"^\s*(\d+\.?\s+)?(methods?|background|related\s+work|literature\s+review"
    r"|materials?\s+and\s+methods?|preliminary|overview)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_introduction_section(full_text: str) -> str:
    """
    Find the introduction header and return text until the next section header
    or 2000 words, whichever comes first. Falls back to first 2000 words.
    """
    intro_match = _INTRO_HEADERS.search(full_text)
    if not intro_match:
        # Fallback: first 2000 words
        return " ".join(full_text.split()[:2000])

    body_start = intro_match.end()
    body = full_text[body_start:]

    next_match = _NEXT_SECTION_HEADERS.search(body)
    if next_match:
        section_text = body[: next_match.start()]
    else:
        section_text = body

    # Enforce 2000-word cap
    words = section_text.split()
    return " ".join(words[:2000])


def pdf_to_audit_text(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Return (full_text, introduction_section) from raw PDF bytes.
    Never raises — errors are embedded in the returned strings.
    """
    full_text = extract_text_from_pdf(pdf_bytes)
    introduction = extract_introduction_section(full_text)
    return full_text, introduction
