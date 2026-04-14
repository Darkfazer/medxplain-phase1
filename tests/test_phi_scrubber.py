"""
tests/test_phi_scrubber.py
===========================
Unit tests for data_ingestion.phi_scrubber.

All tests are self-contained: no real DICOM files or PHI required.
pydicom Datasets are built in-memory.

Run with:
    pytest tests/test_phi_scrubber.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# pydicom availability guard
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    _PYDICOM = True
except ImportError:
    _PYDICOM = False

pytestmark = pytest.mark.skipif(not _PYDICOM, reason="pydicom not installed")

import numpy as np
import torch

from data_ingestion.phi_scrubber import (
    DICOM_CONFIDENTIALITY_KEYWORDS,
    SAFE_DICOM_KEYWORDS,
    scrub_report_text,
    _pseudonymise_salt,
    _scrub_metadata,
    _hash_source_paths,
)
from data_ingestion.dicom_pipeline import DICOMStudy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_study(include_report: bool = True) -> DICOMStudy:
    """Build a DICOMStudy that simulates raw hospital data (before scrubbing)."""
    return DICOMStudy(
        patient_id  = _pseudonymise_salt("PT-9999"),   # already pseudo from loader
        study_uid   = _pseudonymise_salt("UID-0001"),
        modality    = "CR",
        images      = [torch.zeros(1, 64, 64)],
        metadata    = {
            "modality":          "CR",
            "rows":              512,
            "columns":           512,
            "window_center":     200.0,
            "window_width":      400.0,
            "study_description": "Chest PA Dr. Smith",  # potentially identifying
        },
        report_text = (
            "CLINICAL HISTORY:\n65yo male, MRN 12345678, referred by Dr. Johnson.\n\n"
            "FINDINGS:\nSmall right pleural effusion. No pneumothorax.\n\n"
            "IMPRESSION:\nSmall right pleural effusion. Reported by Dr. A. Williams on 04/14/2026."
        ) if include_report else None,
        source_paths = ["/hospital/data/study_001/img001.dcm"],
    )


# ---------------------------------------------------------------------------
# Tests: report text scrubbing
# ---------------------------------------------------------------------------

class TestScrubReportText:
    def test_mrn_redacted(self):
        text   = "Patient MRN 12345678 presented with cough."
        result = scrub_report_text(text, use_ner=False)
        assert "12345678" not in result
        assert "[REDACTED_MRN]" in result

    def test_phone_redacted(self):
        text   = "Contact number: 555-867-5309"
        result = scrub_report_text(text, use_ner=False)
        assert "867-5309" not in result
        assert "[REDACTED_PHONE]" in result

    def test_date_redacted(self):
        text   = "Study performed on 04/14/2026."
        result = scrub_report_text(text, use_ner=False)
        assert "04/14/2026" not in result
        assert "[REDACTED_DATE]" in result

    def test_email_redacted(self):
        text   = "Send results to doctor@hospital.com"
        result = scrub_report_text(text, use_ner=False)
        assert "doctor@hospital.com" not in result
        assert "[REDACTED_EMAIL]" in result

    def test_physician_name_redacted(self):
        text   = "Reported by Dr. Williams."
        result = scrub_report_text(text, use_ner=False)
        # "Dr. Williams" should be caught by the physician pattern
        assert "Williams" not in result or "[REDACTED" in result

    def test_clinical_content_preserved(self):
        """Medical terms that are not PHI must NOT be redacted."""
        text   = "Small right pleural effusion. No pneumothorax."
        result = scrub_report_text(text, use_ner=False)
        assert "pleural effusion" in result.lower()
        assert "pneumothorax" in result.lower()

    def test_empty_string(self):
        assert scrub_report_text("", use_ner=False) == ""

    def test_none_like_empty(self):
        # The function accepts None gracefully (returns as-is)
        result = scrub_report_text(None, use_ner=False)  # type: ignore
        assert result is None


# ---------------------------------------------------------------------------
# Tests: metadata scrubbing
# ---------------------------------------------------------------------------

class TestScrubMetadata:
    def test_study_description_removed(self):
        """study_description may contain physician names – must be removed."""
        raw_meta = {
            "modality":          "CT",
            "rows":              512,
            "study_description": "Chest – Dr. Smith",
        }
        clean = _scrub_metadata(raw_meta)
        assert "study_description" not in clean

    def test_safe_fields_preserved(self):
        """rows, columns, window_* must survive scrubbing."""
        raw_meta = {
            "modality":      "MR",
            "rows":          256,
            "columns":       256,
            "window_center": 100.0,
            "window_width":  500.0,
        }
        clean = _scrub_metadata(raw_meta)
        assert clean["rows"] == 256
        assert clean["columns"] == 256
        assert clean["window_center"] == 100.0

    def test_image_position_removed(self):
        """image_position could be used to identify scanner location."""
        raw_meta = {"modality": "CT", "image_position": [0.0, 0.0, 0.0]}
        clean = _scrub_metadata(raw_meta)
        assert "image_position" not in clean


# ---------------------------------------------------------------------------
# Tests: pseudonymisation
# ---------------------------------------------------------------------------

class TestPseudonymiseSalt:
    def test_deterministic(self):
        a = _pseudonymise_salt("UID-001")
        b = _pseudonymise_salt("UID-001")
        assert a == b

    def test_different_inputs(self):
        assert _pseudonymise_salt("UID-001") != _pseudonymise_salt("UID-002")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MEDXPLAIN_PSEUDO_SALT", "hospital_specific_salt")
        with_custom = _pseudonymise_salt("UID-001")
        monkeypatch.delenv("MEDXPLAIN_PSEUDO_SALT")
        with_default = _pseudonymise_salt("UID-001")
        assert with_custom != with_default


# ---------------------------------------------------------------------------
# Tests: audit hash helper
# ---------------------------------------------------------------------------

class TestHashSourcePaths:
    def test_deterministic(self):
        paths = ["/data/a.dcm", "/data/b.dcm"]
        assert _hash_source_paths(paths) == _hash_source_paths(paths)

    def test_different_paths(self):
        a = _hash_source_paths(["/data/a.dcm"])
        b = _hash_source_paths(["/data/b.dcm"])
        assert a != b

    def test_empty_list(self):
        result = _hash_source_paths([])
        # Should not raise; returns hash of empty string
        assert isinstance(result, str) and len(result) == 64


# ---------------------------------------------------------------------------
# Integration: scrub_dicom_study
# ---------------------------------------------------------------------------

class TestScrubDicomStudy:
    """
    High-level integration tests for the full study scrubbing pipeline.
    Tests run without writing an audit log to disk by overriding the path.
    """

    @pytest.fixture(autouse=True)
    def patch_audit_log(self, tmp_path, monkeypatch):
        """Write audit log to a temp directory."""
        import data_ingestion.phi_scrubber as ps
        monkeypatch.setattr(ps, "_AUDIT_LOG_PATH", tmp_path / "audit.log")

    def test_report_scrubbed(self):
        from data_ingestion.phi_scrubber import scrub_dicom_study
        raw   = _make_raw_study(include_report=True)
        clean = scrub_dicom_study(raw, keep_longitudinal_link=True, use_ner=False)
        assert clean.report_text is not None
        assert "12345678" not in clean.report_text          # MRN gone
        assert "pleural effusion" in clean.report_text.lower()  # finding preserved

    def test_no_source_paths_in_clean(self):
        from data_ingestion.phi_scrubber import scrub_dicom_study
        raw   = _make_raw_study()
        clean = scrub_dicom_study(raw, use_ner=False)
        assert clean.source_paths == []

    def test_images_preserved(self):
        from data_ingestion.phi_scrubber import scrub_dicom_study
        raw   = _make_raw_study()
        clean = scrub_dicom_study(raw, use_ner=False)
        assert len(clean.images) == len(raw.images)

    def test_longitudinal_link_consistent(self):
        """Same source study → same pseudonymised IDs (deterministic)."""
        from data_ingestion.phi_scrubber import scrub_dicom_study
        raw     = _make_raw_study()
        clean_a = scrub_dicom_study(raw, keep_longitudinal_link=True, use_ner=False)
        clean_b = scrub_dicom_study(raw, keep_longitudinal_link=True, use_ner=False)
        assert clean_a.patient_id == clean_b.patient_id
        assert clean_a.study_uid  == clean_b.study_uid

    def test_no_report_handling(self):
        from data_ingestion.phi_scrubber import scrub_dicom_study
        raw   = _make_raw_study(include_report=False)
        clean = scrub_dicom_study(raw, use_ner=False)
        assert clean.report_text is None

    def test_study_description_scrubbed_from_metadata(self):
        from data_ingestion.phi_scrubber import scrub_dicom_study
        raw   = _make_raw_study()
        clean = scrub_dicom_study(raw, use_ner=False)
        assert "study_description" not in clean.metadata
