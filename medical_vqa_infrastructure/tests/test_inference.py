from inference.calibration import TemperatureScaling
import torch

def test_temperature_scaling():
    ts = TemperatureScaling(temperature=1.263)
    logits = torch.tensor([[1.0, 2.0]])
    calibrated = ts.calibrate(logits)
    assert torch.allclose(calibrated.sum(), torch.tensor(1.0))
