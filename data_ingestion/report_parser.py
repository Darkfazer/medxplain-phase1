"""
report_parser.py
================
Extract structured clinical information from free-text radiology reports.

Features:
  - Section extraction: clinical_history, findings, impression
  - Pathology detection with presence/absence, laterality, severity
  - Offline / air-gapped (pure Python stdlib + regex, no model required)
  - Extensible finding catalogue

Usage:
    from data_ingestion.report_parser import ReportParser

    parser = ReportParser()
    result = parser.parse(report_text)
    # result["findings"]["pleural_effusion"]["present"] → True
    # result["findings"]["pleural_effusion"]["severity"]  → "moderate"
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section boundary patterns
# ---------------------------------------------------------------------------

# Each entry: (section_key, list_of_header_patterns)
# Patterns are case-insensitive and match the section heading.
_SECTION_HEADERS: List[Tuple[str, List[str]]] = [
    ("clinical_history", [
        r"clinical\s+(?:history|indication|information)",
        r"indication",
        r"reason\s+for\s+(?:study|exam(?:ination)?)",
        r"history",
    ]),
    ("technique", [
        r"technique",
        r"protocol",
        r"procedure",
    ]),
    ("comparison", [
        r"comparison",
        r"prior\s+(?:study|exam(?:ination)?|imaging)",
    ]),
    ("findings", [
        r"findings?",
        r"observations?",
        r"report",
    ]),
    ("impression", [
        r"impression",
        r"conclusion",
        r"summary",
        r"assessment",
        r"diagnosis",
    ]),
]

# Build a compiled pattern to identify any known header line
_ANY_HEADER_PAT = re.compile(
    r"^\s*(?:" + "|".join(
        h for _, headers in _SECTION_HEADERS for h in headers
    ) + r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Pathology catalogue
# ---------------------------------------------------------------------------

# Each entry:
#   (internal_key, display_name, list_of_detection_patterns)
#
# Detection patterns are applied to the "findings" + "impression" blocks.
# The first match determines presence; negation heuristics are applied separately.

PATHOLOGY_CATALOGUE: List[Tuple[str, str, List[str]]] = [
    ("pneumothorax",      "Pneumothorax",      ["pneumothorax", "ptx"]),
    ("pleural_effusion",  "Pleural Effusion",   ["pleural\s+effusion", "hydrothorax"]),
    ("pneumonia",         "Pneumonia",          ["pneumon(?:ia|itis)", "consolidat(?:ion|ed)"]),
    ("atelectasis",       "Atelectasis",        ["atelectasis", "atelectatic", "collapse(?:d)?"]),
    ("cardiomegaly",      "Cardiomegaly",       ["cardiomegaly", "enlarged\s+(?:cardiac\s+)?silhouette", "cardiac\s+enlargement"]),
    ("pulmonary_edema",   "Pulmonary Edema",    ["pulmonary\s+(?:oedema|edema)", "vascular\s+congestion", "vascular\s+engorgement"]),
    ("nodule",            "Pulmonary Nodule",   [r"(?:pulmonary\s+)?nodule", r"mass\s+lesion", r"rounded\s+opacity"]),
    ("fracture",          "Fracture",           ["fracture", "break", "cortical\s+disruption"]),
    ("pneumoperitoneum",  "Pneumoperitoneum",   ["pneumoperitoneum", "free\s+air", "free\s+gas"]),
    ("pleural_thickening","Pleural Thickening", ["pleural\s+thick(?:en(?:ing)?)?", "pleural\s+calcif"]),
    ("pleural_mass",      "Pleural Mass",       ["pleural\s+(?:mass|lesion|tumou?r)"]),
    ("hilar_enlargement", "Hilar Enlargement",  ["hilar\s+(?:enlarge|prominent)", "enlarged\s+hila"]),
    ("interstitial",      "Interstitial Pattern",["interstitial\s+(?:marking|pattern|opacit)", "reticul(?:ar|onodular)"]),
    ("aortic_widening",   "Aortic Widening",    ["aortic\s+(?:widen|enlarg|dilation|aneurysm)"]),
    ("pacemaker",         "Pacemaker / Device", ["pacemaker", "ICD", "cardiac\s+device", "leads?\s+terminating"]),
]

# Negation triggers that, if found immediately before the finding keyword,
# indicate ABSENCE of the finding.  Window is up to 8 words before the match.
NEGATION_TRIGGERS: List[str] = [
    "no", "not", "without", "absence of", "absent", "negative for",
    "no evidence of", "clear of", "free of", "unremarkable",
    "ruled out", "excluded", "cannot", "cannot identify",
]

_NEGATION_PAT = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in NEGATION_TRIGGERS) + r")\b",
    re.IGNORECASE,
)

# Laterality patterns
_LATERALITY_PAT = re.compile(
    r"\b(left|right|bilateral|bibasilar|bilat\.?)\b", re.IGNORECASE
)

# Severity patterns
_SEVERITY_PAT = re.compile(
    r"\b(trace|tiny|small|mild(?:ly)?|minor|moderate(?:ly)?|large|massive|severe(?:ly)?|significant(?:ly)?)\b",
    re.IGNORECASE,
)

# Normal / no acute findings keywords
_NORMAL_PAT = re.compile(
    r"\b(no\s+acute|unremarkable|within\s+normal\s+limits?|normal\s+(?:appearance|study|exam)|no\s+significant)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

class ReportParser:
    """
    Parse free-text radiology reports into a structured dictionary.

    Parameters
    ----------
    pathology_catalogue:
        Override the default ``PATHOLOGY_CATALOGUE`` with a custom list.
    negation_window_words:
        Number of words before a finding keyword to search for negation.
    """

    def __init__(
        self,
        pathology_catalogue: Optional[List] = None,
        negation_window_words: int = 8,
    ) -> None:
        self._catalogue = pathology_catalogue or PATHOLOGY_CATALOGUE
        self._neg_window = negation_window_words

        # Precompile per-finding detection patterns
        self._finding_patterns: Dict[str, re.Pattern] = {}
        for key, _display, patterns in self._catalogue:
            combined = "|".join(f"(?:{p})" for p in patterns)
            self._finding_patterns[key] = re.compile(combined, re.IGNORECASE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, report_text: str) -> Dict[str, Any]:
        """
        Parse a radiology report string.

        Parameters
        ----------
        report_text:
            Raw free-text radiology report.

        Returns
        -------
        Dictionary with keys:
            - ``clinical_history``  (str | None)
            - ``technique``         (str | None)
            - ``comparison``        (str | None)
            - ``findings_raw``      (str | None)  – raw findings block
            - ``impression``        (str | None)
            - ``findings``          (Dict)  – structured finding results
            - ``is_normal``         (bool)  – True if report flags no acute findings
        """
        if not report_text or not report_text.strip():
            logger.warning("Empty report text provided.")
            return self._empty_result()

        sections = self._extract_sections(report_text)

        # The text relevant for pathology detection
        analysis_text = " ".join(filter(None, [
            sections.get("findings"),
            sections.get("impression"),
        ]))

        structured_findings = self._extract_findings(analysis_text)

        return {
            "clinical_history": sections.get("clinical_history"),
            "technique":        sections.get("technique"),
            "comparison":       sections.get("comparison"),
            "findings_raw":     sections.get("findings"),
            "impression":       sections.get("impression"),
            "findings":         structured_findings,
            "is_normal":        bool(_NORMAL_PAT.search(analysis_text)),
        }

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str) -> Dict[str, Optional[str]]:
        """
        Split the report into named sections using known header patterns.

        Strategy
        --------
        1. Find all header matches and their positions.
        2. The text between header_i and header_{i+1} is the content of
           section_i.
        3. Unrecognised preamble before the first header is stored as
           ``preamble`` and used as a fallback ``clinical_history`` if
           none is found.
        """
        sections: Dict[str, Optional[str]] = {}
        lines = text.splitlines()

        # Build an ordered list of (line_idx, section_key) from headers
        tagged: List[Tuple[int, str]] = []

        for i, line in enumerate(lines):
            for section_key, header_patterns in _SECTION_HEADERS:
                for pat_str in header_patterns:
                    if re.match(
                        r"^\s*" + pat_str + r"\s*[:\-]?\s*$",
                        line,
                        re.IGNORECASE,
                    ):
                        tagged.append((i, section_key))
                        break
                else:
                    continue
                break

        if not tagged:
            # No headers found: treat entire text as "findings"
            sections["findings"] = text.strip()
            return sections

        # Preamble before first section
        preamble = "\n".join(lines[: tagged[0][0]]).strip()
        if preamble:
            sections.setdefault("clinical_history", preamble)

        # Extract content between consecutive headers
        for idx, (line_idx, section_key) in enumerate(tagged):
            content_start = line_idx + 1
            content_end   = tagged[idx + 1][0] if idx + 1 < len(tagged) else len(lines)
            content = "\n".join(lines[content_start:content_end]).strip()
            # If the same section appears multiple times, concatenate
            if section_key in sections and sections[section_key]:
                sections[section_key] = sections[section_key] + "\n" + content
            else:
                sections[section_key] = content or None

        return sections

    # ------------------------------------------------------------------
    # Pathology detection
    # ------------------------------------------------------------------

    def _extract_findings(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Identify presence/absence, laterality, and severity for each
        catalogued pathology.

        Parameters
        ----------
        text:
            Combined findings + impression text.

        Returns
        -------
        Dict mapping internal_key → {present, laterality?, severity?}
        """
        results: Dict[str, Dict[str, Any]] = {}

        for key, _display, _patterns in self._catalogue:
            pat = self._finding_patterns[key]

            # Find all matches in text
            matches = list(pat.finditer(text))
            if not matches:
                results[key] = {"present": False}
                continue

            # For each match check for contextual negation
            any_positive = False
            best_laterality: Optional[str] = None
            best_severity: Optional[str] = None

            for m in matches:
                # Extract a window of text before the match
                window_start = max(0, m.start() - 120)
                pre_window = text[window_start: m.start()]
                # Rough word count limit
                pre_words = pre_window.split()
                pre_context = " ".join(pre_words[-self._neg_window :])

                is_negated = bool(_NEGATION_PAT.search(pre_context))
                if is_negated:
                    continue

                any_positive = True

                # Extract a window around the match for qualifiers
                qualifier_window = text[max(0, m.start() - 60): m.end() + 80]

                lat_m = _LATERALITY_PAT.search(qualifier_window)
                if lat_m and best_laterality is None:
                    best_laterality = lat_m.group(1).lower()

                sev_m = _SEVERITY_PAT.search(qualifier_window)
                if sev_m and best_severity is None:
                    best_severity = self._normalise_severity(sev_m.group(1).lower())

            finding: Dict[str, Any] = {"present": any_positive}
            if any_positive:
                if best_laterality:
                    finding["laterality"] = best_laterality
                if best_severity:
                    finding["severity"] = best_severity

            results[key] = finding

        return results

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_severity(raw: str) -> str:
        """Map synonymous severity terms to canonical values."""
        mild_synonyms    = {"trace", "tiny", "small", "mild", "mildly", "minor"}
        moderate_synonyms = {"moderate", "moderately"}
        severe_synonyms   = {"large", "massive", "severe", "severely", "significant", "significantly"}

        raw_lower = raw.lower()
        if raw_lower in mild_synonyms:
            return "mild"
        if raw_lower in moderate_synonyms:
            return "moderate"
        if raw_lower in severe_synonyms:
            return "severe"
        return raw_lower

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "clinical_history": None,
            "technique":        None,
            "comparison":       None,
            "findings_raw":     None,
            "impression":       None,
            "findings":         {},
            "is_normal":        False,
        }

    def parse_batch(self, reports: List[str]) -> List[Dict[str, Any]]:
        """Parse a list of report strings and return a list of result dicts."""
        return [self.parse(r) for r in reports]

    def extract_impression(self, report_text: str) -> Optional[str]:
        """Convenience shortcut to get only the impression section."""
        return self.parse(report_text).get("impression")

    def is_finding_present(self, report_text: str, finding_key: str) -> bool:
        """
        Quick boolean check for a single pathology.

        Parameters
        ----------
        finding_key:
            One of the internal keys in ``PATHOLOGY_CATALOGUE``
            (e.g. ``"pneumothorax"``).
        """
        result = self.parse(report_text)
        return result["findings"].get(finding_key, {}).get("present", False)
