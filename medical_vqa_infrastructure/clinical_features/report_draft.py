class ReportDrafter:
    """Generates a structured medical report."""
    def generate_draft(self, findings: list) -> str:
        draft = "FINDINGS:\n"
        for f in findings:
            draft += f"- {f}\n"
        draft += "\nIMPRESSION: Refer to findings above."
        return draft
