from config import Config
from model_inference import MedicalVQAModel
import numpy as np

class AdvancedClinicalFeatures:
    def __init__(self, vqa_model: MedicalVQAModel):
        self.vqa_model = vqa_model

    def report_aware_answering(self, image, previous_report, question):
        """
        a) Report-Aware: Combine prior report + current image.
        """
        # Inject the previous report into the systemic prompt for the LLM
        augmented_question = f"Previous Report: {previous_report}\n\nBased on the current image and previous report, {question}"
        answer, conf = self.vqa_model.generate_answer(image, augmented_question)
        return answer, conf

    def context_aware_answering(self, image, clinical_context, question):
        """
        b) Context-Aware: Use clinical notes, labs, vitals
        """
        augmented_question = f"Clinical Context (Labs/Vitals): {clinical_context}\n\nQuestion: {question}"
        answer, conf = self.vqa_model.generate_answer(image, augmented_question)
        return answer, conf
        
    def generate_one_click_report(self, image):
        """
        c) One-Click Report: Draft a structured radiology report.
        """
        # We query the VQA model with standardized structured prompts
        findings_prompt = "Describe the physiological findings in this chest x-ray clearly."
        impression_prompt = "Based on the findings, what is your singular overall impression or diagnosis?"
        
        findings, _ = self.vqa_model.generate_answer(image, findings_prompt)
        impression, _ = self.vqa_model.generate_answer(image, impression_prompt)
        
        report = f"RADIOLOGY REPORT DRAFT\n\nFINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}\n\nELECTRONICALLY SIGNED BY:\n[Clinician Name]"
        return report

    def compare_longitudinal(self, current_image, prior_image, question):
        """
        d) Longitudinal View: Compare current and prior images. 
        """
        if self.vqa_model.use_mock:
            return "Based on the longitudinal analysis, the nodule has remained stable in size, but there is newly developed slight effusion.", 0.85
            
        # TODO: Extract features from both images and fuse them
        # f_curr = self.vqa_model.vision_encoder(current_image)
        # f_prior = self.vqa_model.vision_encoder(prior_image)
        # f_diff = f_curr - f_prior # or concat
        # response = self.vqa_model.llm.generate(f_diff, prompt=question)
        return "Difference analysis not implemented yet in real weights.", 0.0

    def differential_diagnosis(self, image, question):
        """
        e) Differential Diagnosis: Return top-3 ranked possible diagnoses.
        """
        if self.vqa_model.use_mock:
            top_3 = {
                "Pneumonia": "Consolidation observed in the lower left lobe suggests Community Acquired Pneumonia.",
                "Atelectasis": "Volume loss in the adjacent segments strongly correlates with atelectasis.",
                "Pleural Effusion": "Blunting of the costophrenic angle."
            }
            return top_3
            
        # TODO: Iterate over labels, gather logits or ask LLM to format as top 3 structured.
        # "Please provide the top 3 differential diagnoses along with brief explanations."
        return {"None": "Not implemented"}
