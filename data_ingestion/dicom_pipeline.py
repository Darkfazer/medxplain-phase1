"""
dicom_pipeline.py
=================
Production-grade DICOM ingestion pipeline for hospital data.

Handles:
  - Single-frame (CR, DX, XR) and multi-frame (CT, MR) DICOM files
  - Common & compressed transfer syntaxes via pydicom pixel handlers
  - Metadata extraction (pseudonymized) into a structured DICOMStudy dataclass
  - Pixel normalisation to float32 tensors suitable for existing vision encoders
  - Graceful error handling for corrupted / incomplete files

Dependencies:
    pip install pydicom pylibjpeg pillow torch numpy

Usage:
    from data_ingestion.dicom_pipeline import DICOMLoader, DICOMStudy
    loader = DICOMLoader()
    study  = loader.load_study("/hospital/data/study_001/")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports – wrapped so the module can be imported in
# environments where pydicom / PIL are not yet installed.
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _PYDICOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDICOM_AVAILABLE = False
    logger.warning(
        "pydicom not installed. Install with: pip install pydicom pylibjpeg"
    )

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DICOM Transfer Syntax UIDs that can be decoded natively by pydicom
SUPPORTED_TRANSFER_SYNTAXES: set = {
    "1.2.840.10008.1.2",      # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",    # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",    # Explicit VR Big Endian
    "1.2.840.10008.1.2.4.50", # JPEG Baseline (requires pylibjpeg)
    "1.2.840.10008.1.2.4.51", # JPEG Extended
    "1.2.840.10008.1.2.4.57", # JPEG Lossless
    "1.2.840.10008.1.2.4.70", # JPEG Lossless, First-Order Selection
    "1.2.840.10008.1.2.4.80", # JPEG-LS Lossless (requires pylibjpeg)
    "1.2.840.10008.1.2.4.81", # JPEG-LS Near-Lossless
    "1.2.840.10008.1.2.4.90", # JPEG 2000 (requires pylibjpeg)
    "1.2.840.10008.1.2.4.91", # JPEG 2000 Part 2
}

# Tags whose raw value will always be included in metadata
SAFE_METADATA_TAGS: Dict[str, str] = {
    "Modality":              "(0008,0060)",
    "StudyDescription":      "(0008,1030)",
    "SeriesDescription":     "(0008,103E)",
    "ImagePositionPatient":  "(0020,0032)",
    "SliceThickness":        "(0018,0050)",
    "WindowCenter":          "(0028,1050)",
    "WindowWidth":           "(0028,1051)",
    "Rows":                  "(0028,0010)",
    "Columns":               "(0028,0011)",
    "PixelSpacing":          "(0028,0030)",
    "BitsAllocated":         "(0028,0100)",
    "NumberOfFrames":        "(0028,0008)",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DICOMStudy:
    """
    A de-identified, tensor-ready representation of one DICOM study.

    Attributes
    ----------
    patient_id:
        SHA-256 pseudonymised patient identifier.  The raw ID is never stored.
    study_uid:
        SHA-256 pseudonymised StudyInstanceUID.  Consistent across all
        files belonging to the same study so longitudinal linkage is
        preserved without exposing the original UID.
    modality:
        DICOM Modality string (e.g. "CR", "CT", "MR").
    images:
        List of float32 tensors with shape ``(C, H, W)`` (C=1 for greyscale,
        C=3 for colour) normalised to ``[0, 1]``.  Each element corresponds
        to one DICOM frame / slice.
    metadata:
        Safe, non-identifying metadata dictionary extracted from DICOM tags.
    raw_pixel_shapes:
        Original ``(rows, cols)`` before any resizing, kept for debugging.
    report_text:
        Optional free-text radiology report attached to the study.
    source_paths:
        Absolute paths of the DICOM files that were loaded (for audit).
    """

    patient_id: str
    study_uid: str
    modality: str
    images: List[torch.Tensor] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_pixel_shapes: List[Tuple[int, int]] = field(default_factory=list)
    report_text: Optional[str] = None
    source_paths: List[str] = field(default_factory=list)

    def to_json_summary(self) -> str:
        """Return a JSON string of non-tensor fields (useful for logging)."""
        summary = {
            "patient_id":  self.patient_id,
            "study_uid":   self.study_uid,
            "modality":    self.modality,
            "n_frames":    len(self.images),
            "metadata":    self.metadata,
            "report_text": self.report_text,
        }
        return json.dumps(summary, indent=2, default=str)


# ---------------------------------------------------------------------------
# Pseudonymisation helpers
# ---------------------------------------------------------------------------

def _pseudonymise(value: str, salt: str = "medxplain_salt_v1") -> str:
    """
    Return a deterministic SHA-256 hex-digest of ``value`` mixed with a
    static ``salt``.  This preserves longitudinal linkage (same patient →
    same pseudo-ID) without reversibly exposing PII.

    Parameters
    ----------
    value:
        Raw identifying string (e.g. PatientID, StudyInstanceUID).
    salt:
        Application-level salt.  Override via the ``MEDXPLAIN_PSEUDO_SALT``
        environment variable in production.
    """
    salt = os.environ.get("MEDXPLAIN_PSEUDO_SALT", salt)
    raw = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]  # 32 hex chars = 128 bits


# ---------------------------------------------------------------------------
# Pixel normalisation
# ---------------------------------------------------------------------------

def _apply_window(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Apply VOI windowing to a raw DICOM pixel array and return float32 [0, 1].

    Standard radiological windowing formula:
        out = clip((px - (WC - WW/2)) / WW + 0.5, 0, 1)
    """
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0
    windowed = np.clip(pixel_array.astype(np.float32), lower, upper)
    return (windowed - lower) / (upper - lower + 1e-8)


def normalise_pixel_array(
    ds: "pydicom.Dataset",
    target_dtype: str = "float32",
) -> np.ndarray:
    """
    Convert a DICOM dataset's pixel data to a normalised numpy array.

    Strategy
    --------
    1. Apply pydicom VOI LUT if present (handles MONOCHROME1/2, windowing).
    2. If WindowCenter/Width tags exist, apply explicit windowing.
    3. Min-max normalise to ``[0, 1]`` as fallback.
    4. Expand greyscale to ``(1, H, W)`` or keep ``(3, H, W)`` for colour.

    Returns
    -------
    np.ndarray with shape ``(C, H, W)`` and dtype ``float32``.
    """
    try:
        pixel_array = ds.pixel_array  # pydicom decodes compressed data here
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode pixel data: {exc}. "
            "Ensure pylibjpeg or gdcm is installed for compressed syntaxes."
        ) from exc

    # Ensure float computations
    pixel_array = pixel_array.astype(np.float32)

    # --- Handle multi-frame: shape becomes (N, H, W) or (N, H, W, C) ---
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, ...]  # (1, H, W)
    elif pixel_array.ndim == 3:
        if pixel_array.shape[-1] in (3, 4):
            # Colour image (H, W, C) → (C, H, W)
            pixel_array = np.moveaxis(pixel_array, -1, 0)
            pixel_array = pixel_array[:3, ...]  # drop alpha if present
        else:
            # Multi-frame greyscale (N, H, W) – keep as is, handled below
            pass

    # --- Apply VOI LUT / windowing per frame ---
    has_window = hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth")

    def _normalise_frame(frame: np.ndarray) -> np.ndarray:
        if has_window:
            wc = float(
                ds.WindowCenter[0]
                if hasattr(ds.WindowCenter, "__iter__")
                else ds.WindowCenter
            )
            ww = float(
                ds.WindowWidth[0]
                if hasattr(ds.WindowWidth, "__iter__")
                else ds.WindowWidth
            )
            return _apply_window(frame, wc, ww)
        else:
            # Min-max normalise
            mn, mx = frame.min(), frame.max()
            if mx > mn:
                return (frame - mn) / (mx - mn)
            return np.zeros_like(frame)

    if pixel_array.ndim == 3 and pixel_array.shape[0] not in (1, 3):
        # True multi-frame: (N, H, W) → normalise each frame
        normalised = np.stack(
            [_normalise_frame(pixel_array[i]) for i in range(pixel_array.shape[0])],
            axis=0,
        )
        # Return as (N, 1, H, W) — caller will split into list
        return normalised[:, np.newaxis, :, :]
    else:
        # (1, H, W) greyscale or (3, H, W) colour
        if pixel_array.shape[0] == 1:
            return _normalise_frame(pixel_array[0])[np.newaxis, ...]
        else:
            # Per-channel normalisation for colour
            return np.stack(
                [_normalise_frame(pixel_array[c]) for c in range(pixel_array.shape[0])],
                axis=0,
            )


def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a ``(C, H, W)`` float32 numpy array to a torch.Tensor."""
    return torch.from_numpy(array.copy()).float()


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(ds: "pydicom.Dataset", source_path: str) -> Dict[str, Any]:
    """
    Extract safe, non-identifying DICOM tags into a plain dictionary.

    Parameters
    ----------
    ds:
        pydicom Dataset.
    source_path:
        Used only for logging; **not** stored in the returned dict.

    Returns
    -------
    Dict with keys matching ``SAFE_METADATA_TAGS`` where available.
    """
    meta: Dict[str, Any] = {}

    tag_map = {
        "Modality":              "modality",
        "StudyDescription":      "study_description",
        "SeriesDescription":     "series_description",
        "ImagePositionPatient":  "image_position",
        "SliceThickness":        "slice_thickness_mm",
        "WindowCenter":          "window_center",
        "WindowWidth":           "window_width",
        "Rows":                  "rows",
        "Columns":               "columns",
        "PixelSpacing":          "pixel_spacing_mm",
        "BitsAllocated":         "bits_allocated",
        "NumberOfFrames":        "number_of_frames",
    }

    for dicom_attr, meta_key in tag_map.items():
        try:
            val = getattr(ds, dicom_attr, None)
            if val is not None:
                # pydicom DSfloat / IS / etc → native Python types
                if hasattr(val, "real"):
                    val = float(val)
                elif hasattr(val, "__iter__") and not isinstance(val, str):
                    val = list(val)
                meta[meta_key] = val
        except Exception as tag_exc:
            logger.debug(
                "Could not read tag %s from %s: %s",
                dicom_attr,
                source_path,
                tag_exc,
            )

    return meta


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class DICOMLoader:
    """
    Loads one or more DICOM files from a directory or file path into a
    ``DICOMStudy`` dataclass.

    Parameters
    ----------
    target_size:
        Optional ``(width, height)`` to resize images before tensor
        conversion.  ``None`` preserves original resolution.
    require_pixel_data:
        If ``True`` (default), files without valid pixel data are skipped
        rather than raising.
    verbose:
        Enable per-file debug logging.
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        require_pixel_data: bool = True,
        verbose: bool = False,
    ) -> None:
        if not _PYDICOM_AVAILABLE:
            raise ImportError(
                "pydicom is required for DICOMLoader. "
                "Install with: pip install pydicom pylibjpeg"
            )
        self.target_size = target_size
        self.require_pixel_data = require_pixel_data
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> Optional["pydicom.Dataset"]:
        """
        Load a single DICOM file.

        Returns
        -------
        pydicom.Dataset or ``None`` if the file is corrupted / not DICOM.
        """
        path = str(path)
        try:
            ds = pydicom.dcmread(path, force=False)
            self._check_transfer_syntax(ds, path)
            return ds
        except InvalidDicomError:
            logger.warning("Skipping non-DICOM file: %s", path)
            return None
        except FileNotFoundError:
            logger.error("File not found: %s", path)
            return None
        except Exception as exc:
            logger.error("Unexpected error loading %s: %s", path, exc)
            return None

    def load_study(
        self,
        study_path: str | Path,
        report_text: Optional[str] = None,
    ) -> Optional[DICOMStudy]:
        """
        Recursively load all DICOM files under ``study_path`` and aggregate
        them into a single ``DICOMStudy``.

        Parameters
        ----------
        study_path:
            Path to a directory containing DICOM files (may be nested) or
            a single ``.dcm`` file.
        report_text:
            Optional radiology report string to attach to the study.

        Returns
        -------
        DICOMStudy or ``None`` if no valid DICOM files were found.
        """
        study_path = Path(study_path)

        # Collect candidate paths
        if study_path.is_file():
            candidate_paths = [study_path]
        elif study_path.is_dir():
            candidate_paths = sorted(study_path.rglob("*"))
        else:
            logger.error("Path does not exist: %s", study_path)
            return None

        all_datasets: List[Tuple[pydicom.Dataset, str]] = []
        for fp in candidate_paths:
            if fp.is_file():
                ds = self.load_file(fp)
                if ds is not None:
                    all_datasets.append((ds, str(fp)))

        if not all_datasets:
            logger.warning("No valid DICOM files found under: %s", study_path)
            return None

        # Use the first successfully loaded dataset as the reference for
        # study-level metadata (PatientID, StudyInstanceUID, modality)
        ref_ds, ref_path = all_datasets[0]
        patient_id_raw = str(getattr(ref_ds, "PatientID", "UNKNOWN"))
        study_uid_raw  = str(getattr(ref_ds, "StudyInstanceUID", "UNKNOWN"))
        modality       = str(getattr(ref_ds, "Modality", "UNKNOWN"))

        study = DICOMStudy(
            patient_id  = _pseudonymise(patient_id_raw),
            study_uid   = _pseudonymise(study_uid_raw),
            modality    = modality,
            metadata    = extract_metadata(ref_ds, ref_path),
            report_text = report_text,
        )

        # Process pixel data for each file
        for ds, fpath in all_datasets:
            tensors, shapes = self._process_pixel_data(ds, fpath)
            study.images.extend(tensors)
            study.raw_pixel_shapes.extend(shapes)
            study.source_paths.append(fpath)

        logger.info(
            "Loaded DICOMStudy: patient=%s, modality=%s, frames=%d, files=%d",
            study.patient_id,
            study.modality,
            len(study.images),
            len(study.source_paths),
        )
        return study

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_transfer_syntax(
        self, ds: "pydicom.Dataset", path: str
    ) -> None:
        """Warn (not raise) if the transfer syntax is not in the known set."""
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if ts is not None and str(ts) not in SUPPORTED_TRANSFER_SYNTAXES:
            logger.warning(
                "Potentially unsupported transfer syntax %s in %s. "
                "Attempting decode anyway; install pylibjpeg or gdcm if this fails.",
                ts,
                path,
            )

    def _process_pixel_data(
        self,
        ds: "pydicom.Dataset",
        fpath: str,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        Normalise pixel data and return a list of tensors (one per frame)
        and original ``(rows, cols)`` shapes.
        """
        tensors: List[torch.Tensor] = []
        shapes:  List[Tuple[int, int]] = []

        try:
            normalised = normalise_pixel_array(ds)
        except RuntimeError as exc:
            if self.require_pixel_data:
                logger.warning("Skipping %s – pixel decode failed: %s", fpath, exc)
            return tensors, shapes

        # Shape is either (C, H, W) single-frame or (N, 1, H, W) multi-frame
        if normalised.ndim == 4:
            # Multi-frame: split along axis 0
            for i in range(normalised.shape[0]):
                frame = normalised[i]  # (1, H, W)
                h, w = frame.shape[1], frame.shape[2]
                shapes.append((h, w))
                if self.target_size:
                    frame = self._resize_frame(frame, self.target_size)
                tensors.append(array_to_tensor(frame))
        else:
            # Single-frame: (C, H, W)
            h, w = normalised.shape[1], normalised.shape[2]
            shapes.append((h, w))
            if self.target_size:
                normalised = self._resize_frame(normalised, self.target_size)
            tensors.append(array_to_tensor(normalised))

        return tensors, shapes

    @staticmethod
    def _resize_frame(
        frame: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize a ``(C, H, W)`` float32 frame to ``(C, target_h, target_w)``
        using PIL for high-quality resampling.
        """
        if not _PIL_AVAILABLE:
            logger.warning("PIL not available; skipping resize.")
            return frame

        c = frame.shape[0]
        out_channels = []
        for ch in range(c):
            ch_img = (frame[ch] * 255.0).astype(np.uint8)
            pil_img = PILImage.fromarray(ch_img, mode="L")
            pil_img = pil_img.resize(target_size, PILImage.LANCZOS)
            out_channels.append(np.array(pil_img, dtype=np.float32) / 255.0)
        return np.stack(out_channels, axis=0)
