import torch
from models.vqa_model import MedicalVQAModel

def test_vqa_model_mock():
    model = MedicalVQAModel(config={}, mock_mode=True)
    dummy_img = torch.rand(2, 3, 224, 224)
    dummy_q = ["Is this normal?", "Findings?"]
    
    out = model(dummy_img, dummy_q)
    assert out.shape[0] == 2
