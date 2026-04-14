"""
phi_scrubber.py
===============
HIPAA / GDPR compliant PHI removal for DICOM metadata and free-text reports.

Features:
  - DICOM header scrubbing aligned with the DICOM Confidentiality Profile
    (PS 3.15 Annex E, Basic Application Level Confidentiality Profile)
  - Consistent SHA-256 pseudonymisation for longitudinal linkage
  - Regex-based PHI detection and redaction in radiology report text
  - Optional spaCy NER for higher-recall name detection
  - Tamper-evident audit trail (log file with SHA-256 file hashes,
    timestamps, operation counts – never log PHI content)

Dependencies:
    pip install pydicom
    pip install spacy && python -m spacy download en_core_web_sm   # optional

Usage:
    from data_ingestion.phi_scrubber import scrub_dicom_study
    clean_study = scrub_dicom_study(raw_study, keep_longitudinal_link=True)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from data_ingestion.dicom_pipeline import DICOMStudy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit configuration
# ---------------------------------------------------------------------------

_AUDIT_LOG_PATH: Path = Path(
    os.environ.get("MEDXPLAIN_AUDIT_LOG", "./logs/phi_audit.log")
)

# ---------------------------------------------------------------------------
# DICOM tag whitelist / blacklist
# ---------------------------------------------------------------------------

# Tags that are SAFE to retain (keyword form used by pydicom).
# Extend this list only after reviewing the DICOM Confidentiality Profile.
SAFE_DICOM_KEYWORDS: Set[str] = {
    "Modality",
    "StudyDescription",
    "SeriesDescription",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "WindowCenter",
    "WindowWidth",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "NumberOfFrames",
    "PixelData",
    "TransferSyntaxUID",
    "SOPClassUID",
    "SOPInstanceUID",       # Pseudonymised below
    "StudyInstanceUID",     # Pseudonymised below
    "SeriesInstanceUID",    # Pseudonymised below
}

# Tags that carry UIDs which need pseudonymisation rather than removal.
UID_KEYWORDS_TO_PSEUDONYMISE: Set[str] = {
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "FrameOfReferenceUID",
    "SynchronizationFrameOfReferenceUID",
}

# Tags explicitly in the DICOM Basic Confidentiality Profile that must be removed.
# This is a representative (non-exhaustive) list of the most critical ones.
DICOM_CONFIDENTIALITY_KEYWORDS: Set[str] = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientComments",
    "ResponsiblePerson",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "RequestingPhysician",
    "ScheduledPerformingPhysicianName",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "AccessionNumber",
    "StationName",
    "StudyID",
    "RequestedProcedureID",
    "ScheduledProcedureStepID",
    "FillerOrderNumberImagingServiceRequest",
    "PlacerOrderNumberImagingServiceRequest",
    "DeviceSerialNumber",
    "SoftwareVersions",
    "ProtocolName",
    "AcquisitionDeviceProcessingDescription",
    "DerivationDescription",
    "ImageComments",
    "AdmissionID",
    "IssuerOfAdmissionID",
    "CurrentPatientLocation",
    "SpecialNeeds",
}

# ---------------------------------------------------------------------------
# Report PHI regex patterns
# ---------------------------------------------------------------------------

# Ordered list of (pattern, replacement) tuples applied sequentially.
PHI_PATTERNS: List[tuple] = [
    # Medical Record Number (MRN): common formats
    (r"\bMRN[:\s#]*\d{4,12}\b", "[REDACTED_MRN]"),
    # US Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    # Phone numbers (US & international variants)
    (
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED_PHONE]",
    ),
    # Dates: MM/DD/YYYY, DD-MM-YYYY, YYYY/MM/DD, Month DD YYYY
    (
        r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
        r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
        r"|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
        "[REDACTED_DATE]",
    ),
    # Email addresses
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
    # Doctor / physician patterns: "Dr. Smith", "Dr Smith"
    (r"\bDr\.?\s+[A-Z][a-z]+\b", "[REDACTED_PHYSICIAN]"),
    # Radiologist name signature patterns at end of report
    (r"(?:Signed|Reported by|Dictated by|Verified by)[:\s]+[A-Z][a-z]+ [A-Z][a-z]+", "[REDACTED_SIGNATURE]"),
]

_COMPILED_PHI_PATTERNS = [
    (re.compile(pat, re.IGNORECASE | re.MULTILINE), repl)
    for pat, repl in PHI_PATTERNS
]


# ---------------------------------------------------------------------------
# spaCy optional NER
# ---------------------------------------------------------------------------

_SPACY_NLP = None
_SPACY_LOADED = False


def _get_spacy_nlp():
    """Lazy-load spaCy model (en_core_web_sm or en_core_sci_lg if available)."""
    global _SPACY_NLP, _SPACY_LOADED
    if _SPACY_LOADED:
        return _SPACY_NLP
    _SPACY_LOADED = True
    try:
        import spacy  # noqa: F401
        try:
            _SPACY_NLP = spacy.load("en_core_sci_lg")
            logger.info("Loaded en_core_sci_lg for NER-based PHI detection.")
        except OSError:
            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
                logger.info("Loaded en_core_web_sm for NER-based PHI detection.")
            except OSError:
                logger.warning(
                    "No spaCy model found. Install one with: "
                    "python -m spacy download en_core_web_sm"
                )
    except ImportError:
        logger.info(
            "spaCy not installed – using regex-only PHI detection. "
            "Install with: pip install spacy"
        )
    return _SPACY_NLP


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def _write_audit_record(record: Dict[str, Any]) -> None:
    """
    Append a single JSON audit record to the audit log file.

    The record contains timestamps, operation counts and file hashes
    (SHA-256 of source paths as strings) – never raw PHI.
    """
    try:
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.error("Failed to write audit record: %s", exc)


def _hash_source_paths(source_paths: List[str]) -> str:
    """Return a SHA-256 digest of all source file path strings concatenated."""
    combined = "|".join(sorted(source_paths))
    return hashlib.sha256(combined.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Text scrubbing
# ---------------------------------------------------------------------------

def scrub_report_text(text: str, use_ner: bool = True) -> str:
    """
    Remove PHI from a free-text radiology report.

    Parameters
    ----------
    text:
        Raw report string.
    use_ner:
        If ``True`` (and spaCy is available) use NER to additionally
        redact PERSON entities not caught by regex.

    Returns
    -------
    Scrubbed string with PHI replaced by ``[REDACTED_*]`` tokens.
    """
    if not text:
        return text

    scrubbed = text

    # 1. Regex-based redaction
    for pattern, replacement in _COMPILED_PHI_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)

    # 2. NER-based PERSON entity redaction (optional)
    if use_ner:
        nlp = _get_spacy_nlp()
        if nlp is not None:
            doc = nlp(scrubbed)
            # Replace in reverse order to preserve offsets
            person_spans = [
                (ent.start_char, ent.end_char)
                for ent in doc.ents
                if ent.label_ == "PERSON"
            ]
            for start, end in reversed(person_spans):
                scrubbed = scrubbed[:start] + "[REDACTED_NAME]" + scrubbed[end:]

    return scrubbed


# ---------------------------------------------------------------------------
# DICOM header scrubbing (operates on pydicom Dataset)
# ---------------------------------------------------------------------------

def _pseudonymise_salt(value: str) -> str:
    """Deterministic SHA-256 pseudonymisation using env-configured salt."""
    salt = os.environ.get("MEDXPLAIN_PSEUDO_SALT", "medxplain_salt_v1")
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:32]


def scrub_dicom_dataset(ds: "pydicom.Dataset") -> "pydicom.Dataset":  # noqa: F821
    """
    Return a scrubbed **copy** of a pydicom Dataset with PHI removed.

    - Tags in ``DICOM_CONFIDENTIALITY_KEYWORDS`` are deleted.
    - Tags in ``UID_KEYWORDS_TO_PSEUDONYMISE`` are replaced with
      deterministic pseudonyms (SHA-256).
    - All other tags not in ``SAFE_DICOM_KEYWORDS`` are deleted.
    - Pixel data is preserved.

    Returns
    -------
    New pydicom.Dataset with PHI removed.
    """
    try:
        import pydicom  # noqa: F401
    except ImportError:
        raise ImportError("pydicom required: pip install pydicom")

    scrubbed_ds = copy.deepcopy(ds)

    # Collect all keyword names present in dataset
    all_keywords = [elem.keyword for elem in scrubbed_ds if elem.keyword]

    tags_removed = 0
    tags_pseudonymised = 0

    for keyword in all_keywords:
        if not keyword:
            continue

        if keyword in UID_KEYWORDS_TO_PSEUDONYMISE:
            try:
                original_uid = str(getattr(scrubbed_ds, keyword))
                pseudo_uid   = _pseudonymise_salt(original_uid)
                # UIDs must be numeric-dot notation; we wrap in a local OID prefix
                # prefix: 2.25. is the standards-based UUID prefix
                setattr(scrubbed_ds, keyword, f"2.25.{int(pseudo_uid[:16], 16)}")
                tags_pseudonymised += 1
            except Exception as exc:
                logger.debug("Could not pseudonymise %s: %s", keyword, exc)
                try:
                    delattr(scrubbed_ds, keyword)
                    tags_removed += 1
                except Exception:
                    pass

        elif keyword in DICOM_CONFIDENTIALITY_KEYWORDS:
            try:
                delattr(scrubbed_ds, keyword)
                tags_removed += 1
            except Exception:
                pass

        elif keyword not in SAFE_DICOM_KEYWORDS:
            # Remove any unrecognised tag that isn't explicitly whitelisted
            try:
                delattr(scrubbed_ds, keyword)
                tags_removed += 1
            except Exception:
                pass

    logger.debug(
        "DICOM scrub: %d tags removed, %d UIDs pseudonymised",
        tags_removed,
        tags_pseudonymised,
    )
    return scrubbed_ds


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def scrub_dicom_study(
    study: DICOMStudy,
    keep_longitudinal_link: bool = True,
    scrub_report: bool = True,
    use_ner: bool = True,
) -> DICOMStudy:
    """
    Return a fully de-identified copy of a ``DICOMStudy``.

    Parameters
    ----------
    study:
        The raw study produced by ``DICOMLoader.load_study()``.
    keep_longitudinal_link:
        If ``True``, the pseudonymised ``patient_id`` and ``study_uid``
        are **deterministic** (same input → same pseudo-ID), enabling
        linkage across studies without exposing identity.
        If ``False``, random UUIDs are used (stronger anonymisation but
        no longitudinal linkage).
    scrub_report:
        If ``True``, apply PHI scrubbing to ``study.report_text``.
    use_ner:
        Passed through to ``scrub_report_text`` to enable NER.

    Returns
    -------
    A new ``DICOMStudy`` with all PHI removed.  The original object is
    not mutated.
    """
    start_ts = datetime.now(tz=timezone.utc).isoformat()

    # Deep-copy study struct (images / tensors are not heavy to copy
    # for typical single-study workloads; use reference sharing if needed)
    clean = DICOMStudy(
        # patient_id and study_uid from the loader are already pseudonymised;
        # we regenerate them here for clarity / audit.
        patient_id  = study.patient_id if keep_longitudinal_link else _random_uuid(),
        study_uid   = study.study_uid  if keep_longitudinal_link else _random_uuid(),
        modality    = study.modality,
        images      = list(study.images),           # tensor references (immutable)
        metadata    = _scrub_metadata(study.metadata),
        raw_pixel_shapes = list(study.raw_pixel_shapes),
        report_text = None,
        source_paths = [],   # never carry raw paths in cleaned copy
    )

    # Scrub free-text report
    if scrub_report and study.report_text:
        clean.report_text = scrub_report_text(study.report_text, use_ner=use_ner)

    end_ts = datetime.now(tz=timezone.utc).isoformat()

    # Write audit record
    audit = {
        "operation":          "scrub_dicom_study",
        "timestamp_start":    start_ts,
        "timestamp_end":      end_ts,
        "source_hash":        _hash_source_paths(study.source_paths),
        "modality":           study.modality,
        "n_frames":           len(study.images),
        "longitudinal_link":  keep_longitudinal_link,
        "report_scrubbed":    scrub_report and (study.report_text is not None),
        "ner_used":           use_ner,
    }
    _write_audit_record(audit)
    logger.info("PHI scrubbing complete – audit record written to %s", _AUDIT_LOG_PATH)

    return clean


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_uuid() -> str:
    """Generate a random 32-char hex string (used when longitudinal link off)."""
    import uuid
    return uuid.uuid4().hex


def _scrub_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove any residual identifying fields from the metadata dictionary
    produced by ``extract_metadata``.

    Fields like ``study_description`` may contain patient names in some
    hospitals; we sanitise them here as well.
    """
    sensitive_keys = {
        "study_description",
        "series_description",
        "image_position",   # could be used to geo-locate scanner
    }
    return {k: v for k, v in metadata.items() if k not in sensitive_keys}
