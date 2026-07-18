"""Bounded PDF-to-source ingestion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pymupdf

from research_system.config import Settings
from research_system.exceptions import SourceValidationError
from research_system.models import IntegrityLabel, Source, SourceKind
from research_system.tools.base import text_checksum


class PdfParser:
    """Extract a PDF only after enforcing size, encryption, page, and text limits."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def parse(self, data: bytes, filename: str) -> list[Source]:
        if not data:
            raise SourceValidationError("PDF upload is empty")
        if len(data) > self._settings.max_pdf_bytes:
            raise SourceValidationError("PDF exceeds the configured size limit")
        safe_name = Path(filename).name.strip() or "upload.pdf"
        try:
            document: Any = pymupdf.open(  # type: ignore[no-untyped-call]
                stream=data, filetype="pdf"
            )
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise SourceValidationError("Upload is not a readable PDF") from exc
        try:
            if document.needs_pass:
                raise SourceValidationError("encrypted PDFs are not accepted")
            if document.page_count > self._settings.max_pdf_pages:
                raise SourceValidationError("PDF exceeds the configured page limit")
            pages: list[str] = []
            character_count = 0
            effective_character_limit = min(self._settings.max_pdf_characters, 500_000)
            for page in document:
                text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                if not text:
                    continue
                character_count += len(text) + (2 if pages else 0)
                if character_count > effective_character_limit:
                    raise SourceValidationError("PDF exceeds the configured text limit")
                pages.append(text)
        except SourceValidationError:
            raise
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise SourceValidationError("PDF text extraction failed") from exc
        finally:
            document.close()
        content = "\n\n".join(pages)
        if not content:
            raise SourceValidationError("PDF contains no extractable text")
        return [
            Source(
                id="S1",
                kind=SourceKind.PDF,
                title=safe_name[:500],
                url=f"upload://{quote(safe_name)}",
                snippet=content[:20_000],
                content=content,
                provider="PyMuPDF upload parser",
                integrity=IntegrityLabel.USER_UPLOAD,
                locator=f"{len(pages)} extracted page(s)",
                checksum=text_checksum(content),
            )
        ]
