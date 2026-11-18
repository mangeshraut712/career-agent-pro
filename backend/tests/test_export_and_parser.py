"""Unit tests for ExportService and ResumeParser contracts and edge cases."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from schemas import ExportEducation, ExportExperience, ResumeData, ResumeExportData
from security_utils import MAX_EXTRACTED_CHARS, MAX_PDF_PAGES, ResumeUploadError
import services.export_service as export_mod
from services.export_service import ExportService, _export_mapping, escape_latex, pdf_text
from services.resume_parser import ResumeParser


def test_escape_latex_single_pass_no_double_escaping():
    raw = "R&D 50% $100 #1 user_name {a} ~ ^ \\"
    escaped = escape_latex(raw)
    assert r"\&" in escaped
    assert r"\%" in escaped
    assert r"\$" in escaped
    assert r"\#" in escaped
    assert r"\_" in escaped
    assert r"\{a\}" in escaped
    assert r"\textasciitilde{}" in escaped
    assert r"\textasciicircum{}" in escaped
    assert r"\textbackslash{}" in escaped
    # Ensure backslashes inside \textbackslash{} were not re-escaped to \textbackslash{}textbackslash\{\}
    assert r"\textbackslash{}textbackslash" not in escaped
    assert escape_latex(None) == ""


def test_pdf_text_unicode_vs_latin1_fallback():
    assert pdf_text(None, unicode_font=True) == ""
    assert pdf_text("Résumé — café 🚀", unicode_font=True) == "Résumé — café 🚀"
    fallback = pdf_text("Résumé — café 🚀", unicode_font=False)
    assert "Résumé" in fallback
    assert "café" in fallback
    assert "?" in fallback


def test_export_mapping_supports_dicts_and_pydantic_models():
    exp_model = ExportExperience(role="Engineer", company="Acme", duration="2024", description="Built APIs")
    assert _export_mapping({"role": "Lead"}) == {"role": "Lead"}
    assert _export_mapping(exp_model)["company"] == "Acme"
    assert _export_mapping("invalid-entry") is None


def test_get_organized_path_falls_back_when_slug_is_empty(tmp_path):
    out = ExportService.get_organized_path("!!!", base_dir=str(tmp_path))
    assert out.endswith(os.path.join("general"))
    assert os.path.isdir(out)


def test_to_latex_renders_resume_export_data_models_and_escapes():
    export_payload = ResumeExportData(
        name="Ada Lovelace",
        email="ada_lovelace@example.com",
        phone="+1 555-0100",
        linkedin="https://linkedin.com/in/ada",
        summary="Cut latency by 45% & saved $20k.",
        experience=[
            ExportExperience(
                role="Staff Engineer",
                company="Babbage & Co",
                duration="2022--2026",
                description="Scaled #1 engine_v2 by 3x.",
            )
        ],
        education=[
            ExportEducation(
                institution="University of London",
                degree="B.S. Mathematics & Computing",
                graduation_year="2021",
            )
        ],
        skills=["Python", "C++", "Distributed & Cloud"],
    )

    latex = ExportService.to_latex(export_payload)
    assert r"\name{Ada}{Lovelace}" in latex
    assert r"\address{ada\_lovelace@example.com}" in latex
    assert r"\phone[mobile]{+1 555-0100}" in latex
    assert r"\social[linkedin]{https://linkedin.com/in/ada}" in latex
    assert r"Cut latency by 45\% \& saved \$20k." in latex
    assert r"\cventry{2022--2026}{Staff Engineer}{Babbage \& Co}{}{}{Scaled \#1 engine\_v2 by 3x.}" in latex
    assert r"\cventry{2021}{B.S. Mathematics \& Computing}{University of London}{}{}{}" in latex
    assert r"\cvitem{Core Skills}{Python, C++, Distributed \& Cloud}" in latex


def test_to_latex_rejects_blank_name():
    resume = ResumeData(
        name="   ",
        email="a@b.com",
        summary="Summary",
        experience=[],
        education=[],
        skills=[],
    )
    with pytest.raises(ValueError, match="Resume name is required"):
        ExportService.to_latex(resume)


def test_to_docx_and_to_pdf_missing_optional_dependency_raises_runtime_error(tmp_path):
    resume = ResumeData(
        name="Grace Hopper",
        email="grace@example.com",
        summary="Compiler pioneer",
        experience=[],
        education=[],
        skills=["COBOL"],
    )
    with patch.object(export_mod, "Document", None):
        with pytest.raises(RuntimeError, match="python-docx"):
            ExportService.to_docx(resume, str(tmp_path / "out.docx"))

    with patch.object(export_mod, "FPDF", None):
        with pytest.raises(RuntimeError, match="fpdf2"):
            ExportService.to_pdf(resume, str(tmp_path / "out.pdf"))


def test_resume_parser_txt_utf8_latin1_and_truncation():
    content = ("a" * (MAX_EXTRACTED_CHARS + 500)).encode("utf-8")
    extracted = ResumeParser.extract_text(content, "resume.txt")
    assert len(extracted) == MAX_EXTRACTED_CHARS

    latin1_bytes = "José García - Développeur".encode("latin-1")
    decoded = ResumeParser.extract_text(latin1_bytes, "cv.txt")
    assert "Jos" in decoded


def test_resume_parser_rejects_empty_or_unsupported_filename():
    with pytest.raises(ValueError, match="Filename is required"):
        ResumeParser.extract_text(b"hello", "")

    with pytest.raises(ResumeUploadError, match="Unsupported file type"):
        ResumeParser.extract_text(b"hello", "malware.exe")


def test_resume_parser_pdf_page_limit_and_partial_page_failure():
    class FakePageOk:
        def extract_text(self):
            return "Page text ok"

    class FakePageBoom:
        def extract_text(self):
            raise RuntimeError("corrupt glyph table")

    fake_reader_ok = SimpleNamespace(pages=[FakePageBoom(), FakePageOk()])
    with patch("services.resume_parser.PdfReader", return_value=fake_reader_ok):
        assert ResumeParser.extract_text_from_pdf(b"%PDF-fake") == "Page text ok"

    too_many_pages = SimpleNamespace(pages=[FakePageOk()] * (MAX_PDF_PAGES + 1))
    with patch("services.resume_parser.PdfReader", return_value=too_many_pages):
        with pytest.raises(ResumeUploadError, match="page limit"):
            ResumeParser.extract_text_from_pdf(b"%PDF-too-long")
