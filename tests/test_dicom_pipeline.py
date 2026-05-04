"""
tests/test_dicom_pipeline.py
=============================
Unit tests for data_ingestion.dicom_pipeline.

These tests are designed to run WITHOUT real DICOM data:
  - Synthetic pixel arrays replace real scanner output
  - pydicom.Dataset objects are constructed on-the-fly
  - All assertions are against well-defined, deterministic values

Run with:
    pytest tests/test_dicom_pipeline.py -v
"""

import sys
import os

# Ensure project root is in sys.path regardless of where pytest is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Attempt to import; skip entire module gracefully if pydicom is absent
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian
    _PYDICOM = True
except ImportError:
    _PYDICOM = False

pytestmark = pytest.mark.skipif(not _PYDICOM, reason="pydicom not installed")

from data_ingestion.dicom_pipeline import (
    DICOMStudy,
    _pseudonymise,
    array_to_tensor,
    extract_metadata,
    normalise_pixel_array,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic pydicom Datasets
# ---------------------------------------------------------------------------

def _make_greyscale_dataset(rows: int = 64, cols: int = 64,
                             bit_depth: int = 16,
                             window_center: float = 512.0,
                             window_width: float = 1500.0,
                             modality: str = "CR") -> Dataset:
    """Return a minimal single-frame greyscale DICOM Dataset."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.Modality             = modality
    ds.StudyDescription     = "Test Study"
    ds.SeriesDescription    = "Test Series"
    ds.Rows                 = rows
    ds.Columns              = cols
    ds.BitsAllocated        = bit_depth
    ds.BitsStored           = bit_depth
    ds.HighBit              = bit_depth - 1
    ds.PixelRepresentation  = 0
    ds.SamplesPerPixel      = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.WindowCenter         = window_center
    ds.WindowWidth          = window_width
    ds.SliceThickness       = 5.0

    ds.PatientID            = "PAT-12345"
    ds.StudyInstanceUID     = "1.2.3.4.5.6.7.8"
    ds.SeriesInstanceUID    = "1.2.3.4.5.6.7.9"

    # Synthetic pixel data: ramp pattern
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    pixel_array = np.arange(rows * cols, dtype=dtype).reshape(rows, cols)
    ds.PixelData = pixel_array.tobytes()
    ds._pixel_array = pixel_array  # monkey-patch for testing without GDCM

    return ds


def _make_multiframe_dataset(n_frames: int = 4,
                              rows: int = 32,
                              cols: int = 32) -> Dataset:
    """Return a minimal multi-frame CT DICOM Dataset."""
    ds = _make_greyscale_dataset(rows=rows, cols=cols, modality="CT")
    ds.NumberOfFrames = n_frames

    pixel_array = np.random.randint(0, 2000, (n_frames, rows, cols), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()
    ds._pixel_array = pixel_array
    return ds


# Patch pydicom's Dataset.pixel_array to use our pre-baked array
@pytest.fixture(autouse=True)
def patch_pixel_array(monkeypatch):
    """
    Monkeypath pydicom.Dataset.pixel_array so tests don't require
    real pixel data decoding infrastructure (GDCM / pylibjpeg).
    """
    original = Dataset.__class__

    def _pixel_array_prop(self):
        if hasattr(self, "_pixel_array"):
            return self._pixel_array
        raise AttributeError("pixel_array: no _pixel_array stub set")

    monkeypatch.setattr(Dataset, "pixel_array", property(_pixel_array_prop), raising=False)
    yield


# ---------------------------------------------------------------------------
# Tests: pseudonymisation
# ---------------------------------------------------------------------------

class TestPseudonymise:
    def test_deterministic(self):
        """Same input → same output."""
        assert _pseudonymise("PAT-001") == _pseudonymise("PAT-001")

    def test_different_inputs(self):
        """Different inputs → different outputs."""
        assert _pseudonymise("PAT-001") != _pseudonymise("PAT-002")

    def test_length(self):
        """Output is exactly 32 hex characters."""
        assert len(_pseudonymise("PAT-XYZ")) == 32

    def test_salt_env_override(self, monkeypatch):
        """Custom salt via environment variable produces different result."""
        monkeypatch.setenv("MEDXPLAIN_PSEUDO_SALT", "custom_salt")
        result_custom = _pseudonymise("PAT-001")
        monkeypatch.delenv("MEDXPLAIN_PSEUDO_SALT", raising=False)
        result_default = _pseudonymise("PAT-001")
        assert result_custom != result_default


# ---------------------------------------------------------------------------
# Tests: pixel normalisation
# ---------------------------------------------------------------------------

class TestNormalisePixelArray:
    def test_greyscale_shape(self):
        """Single-frame greyscale → shape (1, H, W)."""
        ds = _make_greyscale_dataset(rows=64, cols=64)
        out = normalise_pixel_array(ds)
        assert out.shape == (1, 64, 64), f"Unexpected shape: {out.shape}"

    def test_output_range(self):
        """Pixel values must be clipped to [0, 1]."""
        ds = _make_greyscale_dataset(rows=64, cols=64)
        out = normalise_pixel_array(ds)
        assert float(out.min()) >= 0.0, "Min below 0"
        assert float(out.max()) <= 1.0 + 1e-6, "Max above 1"

    def test_dtype(self):
        """Output dtype must be float32."""
        ds = _make_greyscale_dataset()
        out = normalise_pixel_array(ds)
        assert out.dtype == np.float32

    def test_multiframe_shape(self):
        """Multi-frame CT → shape (N, 1, H, W)."""
        ds = _make_multiframe_dataset(n_frames=4, rows=32, cols=32)
        out = normalise_pixel_array(ds)
        assert out.shape == (4, 1, 32, 32), f"Unexpected multi-frame shape: {out.shape}"

    def test_no_window_fallback(self):
        """When WindowCenter/Width tags are absent, min-max normalisation is used."""
        ds = _make_greyscale_dataset(rows=16, cols=16)
        del ds.WindowCenter
        del ds.WindowWidth
        out = normalise_pixel_array(ds)
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0 + 1e-6

    def test_uniform_image(self):
        """Uniform pixel array should not produce NaN (division by zero guard)."""
        ds = _make_greyscale_dataset(rows=8, cols=8)
        ds._pixel_array = np.zeros((8, 8), dtype=np.uint16)
        del ds.WindowCenter
        del ds.WindowWidth
        out = normalise_pixel_array(ds)
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# Tests: array_to_tensor
# ---------------------------------------------------------------------------

class TestArrayToTensor:
    def test_returns_tensor(self):
        arr = np.zeros((1, 32, 32), dtype=np.float32)
        t   = array_to_tensor(arr)
        assert isinstance(t, torch.Tensor)

    def test_shape_preserved(self):
        arr = np.random.rand(3, 64, 64).astype(np.float32)
        t   = array_to_tensor(arr)
        assert t.shape == torch.Size([3, 64, 64])

    def test_dtype_float32(self):
        arr = np.ones((1, 16, 16), dtype=np.float32)
        t   = array_to_tensor(arr)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: metadata extraction
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    def test_modality_present(self):
        ds   = _make_greyscale_dataset(modality="MR")
        meta = extract_metadata(ds, "/fake/path.dcm")
        assert meta["modality"] == "MR"

    def test_rows_cols(self):
        ds   = _make_greyscale_dataset(rows=128, cols=256)
        meta = extract_metadata(ds, "/fake/path.dcm")
        assert meta["rows"] == 128
        assert meta["columns"] == 256

    def test_window_values(self):
        ds   = _make_greyscale_dataset(window_center=200.0, window_width=400.0)
        meta = extract_metadata(ds, "/fake/path.dcm")
        assert meta["window_center"] == pytest.approx(200.0)
        assert meta["window_width"]  == pytest.approx(400.0)

    def test_missing_tag_skipped(self):
        ds = _make_greyscale_dataset()
        del ds.SliceThickness
        meta = extract_metadata(ds, "/fake/path.dcm")
        assert "slice_thickness_mm" not in meta

    def test_no_patient_data_leaked(self):
        """PatientID must never appear in extracted metadata."""
        ds   = _make_greyscale_dataset()
        meta = extract_metadata(ds, "/fake/path.dcm")
        for key in meta:
            assert "patient" not in key.lower(), (
                f"Potential PHI field found in metadata: '{key}'"
            )


# ---------------------------------------------------------------------------
# Tests: DICOMStudy dataclass
# ---------------------------------------------------------------------------

class TestDICOMStudy:
    def _make_study(self):
        return DICOMStudy(
            patient_id="abc123",
            study_uid="def456",
            modality="CR",
            images=[torch.zeros(1, 64, 64)],
            metadata={"modality": "CR", "rows": 64},
        )

    def test_json_summary_no_exception(self):
        study = self._make_study()
        summary = study.to_json_summary()
        assert "patient_id" in summary
        assert "abc123" in summary

    def test_images_are_tensors(self):
        study = self._make_study()
        assert all(isinstance(t, torch.Tensor) for t in study.images)

    def test_report_text_default_none(self):
        study = self._make_study()
        assert study.report_text is None
