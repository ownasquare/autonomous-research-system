from __future__ import annotations

import pymupdf
import pytest

from research_system.config import Settings
from research_system.exceptions import SourceValidationError
from research_system.models import IntegrityLabel, SourceKind
from research_system.tools.pdf import PdfParser


def _pdf_bytes(text: str = "Grounded research evidence") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_pdf_parser_extracts_a_citable_upload(tmp_path) -> None:
    parser = PdfParser(Settings(research_data_dir=tmp_path))

    sources = parser.parse(_pdf_bytes(), filename="research notes.pdf")

    assert len(sources) == 1
    assert sources[0].kind is SourceKind.PDF
    assert sources[0].integrity is IntegrityLabel.USER_UPLOAD
    assert sources[0].url == "upload://research%20notes.pdf"
    assert "Grounded research evidence" in sources[0].content


def test_pdf_parser_rejects_oversized_upload(tmp_path) -> None:
    parser = PdfParser(Settings(research_data_dir=tmp_path, max_pdf_bytes=100_000))

    with pytest.raises(SourceValidationError, match="size limit"):
        parser.parse(b"%PDF" + b"x" * 100_000, filename="large.pdf")


def test_pdf_parser_rejects_encrypted_upload(tmp_path) -> None:
    document = pymupdf.open(stream=_pdf_bytes(), filetype="pdf")
    encrypted = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="reader-password",
    )
    document.close()
    parser = PdfParser(Settings(research_data_dir=tmp_path))

    with pytest.raises(SourceValidationError, match="encrypted"):
        parser.parse(encrypted, filename="protected.pdf")


def test_pdf_parser_rejects_empty_document(tmp_path) -> None:
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    parser = PdfParser(Settings(research_data_dir=tmp_path))

    with pytest.raises(SourceValidationError, match="no extractable text"):
        parser.parse(data, filename="empty.pdf")


def test_pdf_parser_rejects_page_limit(tmp_path) -> None:
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    data = document.tobytes()
    document.close()
    parser = PdfParser(Settings(research_data_dir=tmp_path, max_pdf_pages=1))

    with pytest.raises(SourceValidationError, match="page limit"):
        parser.parse(data, filename="long.pdf")


def test_pdf_parser_rejects_text_limit(tmp_path) -> None:
    document = pymupdf.open()
    for _ in range(25):
        page = document.new_page()
        page.insert_text((72, 72), "bounded evidence " * 8)
    data = document.tobytes()
    document.close()
    parser = PdfParser(Settings(research_data_dir=tmp_path, max_pdf_characters=1_000))

    with pytest.raises(SourceValidationError, match="text limit"):
        parser.parse(data, filename="verbose.pdf")


def test_pdf_parser_normalizes_page_extraction_failure(monkeypatch, tmp_path) -> None:
    class BrokenPage:
        def get_text(self, _format: str) -> str:
            raise RuntimeError("malformed page object")

    class BrokenDocument:
        needs_pass = False
        page_count = 1

        def __iter__(self):
            yield BrokenPage()

        def close(self) -> None:
            return None

    monkeypatch.setattr(pymupdf, "open", lambda **_kwargs: BrokenDocument())
    parser = PdfParser(Settings(research_data_dir=tmp_path))

    with pytest.raises(SourceValidationError, match="text extraction failed"):
        parser.parse(b"%PDF-broken-page", filename="broken-page.pdf")
