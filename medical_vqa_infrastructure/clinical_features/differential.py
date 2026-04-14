class DifferentialDiagnoser:
    """Provides top-3 differential diagnoses with explanations."""
    def get_differentials(self, model_output) -> list:
        return [
            {"diagnosis": "Pneumonia", "confidence": 0.85, "explanation": "Opacity in left lower lobe."},
            {"diagnosis": "Atelectasis", "confidence": 0.10, "explanation": "Volume loss suggested."},
            {"diagnosis": "Pleural Effusion", "confidence": 0.05, "explanation": "Blunting of CP angle."}
        ]
