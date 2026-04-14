from data.dataset import MedicalVQADataset

def test_dataset_mock():
    dataset = MedicalVQADataset(data_dir="", mock_mode=True)
    assert len(dataset) == 10
    sample = dataset[0]
    assert "image" in sample
    assert "question" in sample
