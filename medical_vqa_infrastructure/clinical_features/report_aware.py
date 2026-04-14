class ReportAwareModule:
    """Concatenates prior report text with the new question."""
    def embed_prior_report(self, question: str, prior_report: str) -> str:
        return f"[PRIOR REPORT]: {prior_report} [QUESTION]: {question}"
