import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
from config import Config
from utils import apply_temperature_scaling, apply_thresholds

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None

class DummyVisionEncoder(nn.Module):
    """A minimal mock vision encoder for testing when real weights are absent."""
    def __init__(self):
        super().__init__()
        # Use simple resnet18 backbone to mock feature extraction
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, Config.NUM_CLASSES)
        
    def forward(self, x):
        return self.backbone(x)

class MedicalVQAModel:
    def __init__(self, use_mock=Config.USE_MOCK_MODEL):
        self.device = Config.DEVICE
        self.use_mock = use_mock
        self.vision_encoder = None
        self.llm = None
        
        # Image Preprocessing mapping exactly to the trained models
        self.transform = transforms.Compose([
            transforms.Resize((Config.MAX_IMAGE_SIZE, Config.MAX_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_models()

    def load_models(self):
        """Loads Vision and Language model weights."""
        if self.use_mock:
            print("[INFO] Loading MOCK models for UI testing.")
            self.vision_encoder = DummyVisionEncoder().to(self.device).eval()
            self.llm = "MOCK_LLM"
        else:
            print("[INFO] Loading real model weights.")
            # TODO: load your trained checkpoint here
            # e.g. self.vision_encoder = MyDenseNet().to(self.device)
            # self.vision_encoder.load_state_dict(torch.load(Config.VISION_ENCODER_PATH))
            
            # TODO: load language model
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            # self.llm = AutoModelForCausalLM.from_pretrained("microsoft/biogpt").to(self.device)
            pass

    def preprocess_image(self, image_input):
        """
        Takes RGB PIL Image or numpy array and returns batched tensor.
        """
        if isinstance(image_input, np.ndarray):
            image_input = Image.fromarray(image_input).convert('RGB')
        else:
            image_input = image_input.convert('RGB')
            
        img_tensor = self.transform(image_input).unsqueeze(0).to(self.device)
        return img_tensor, image_input

    def generate_answer(self, image_input, question: str):
        """
        Main inference for VQA.
        Latency < 5 seconds expected via optimized forward passes.
        """
        img_tensor, _ = self.preprocess_image(image_input)
        
        if self.use_mock:
            import time
            time.sleep(0.5) # simulate latency
            return "Yes, there is evidence of cardiomegaly and patchy infiltrates.", 0.89
            
        # TODO: Real Inference implementation
        # 1. vision_features = self.vision_encoder.extract_features(img_tensor)
        # 2. fused_inputs = fusion_layer(vision_features, question_tokens)
        # 3. out = self.llm.generate(fused_inputs)
        # 4. return self.tokenizer.decode(out), confidence_score
        
        return "Not Implemented", 0.0

    def generate_gradcam(self, image_input, target_class_idx=None):
        """
        Generates a Grad-CAM overlay highlighting the reasoning regions in the image.
        """
        if GradCAM is None:
            return image_input, "GradCAM library not installed."

        img_tensor, orig_img = self.preprocess_image(image_input)
        
        # Identify the target layer for Grad-CAM
        if self.use_mock:
            target_layers = [self.vision_encoder.backbone.layer4[-1]]
            cam = GradCAM(model=self.vision_encoder, target_layers=target_layers, use_cuda=(self.device.type=='cuda'))
        else:
            # TODO: Assign target layers for your specific vision encoder
            target_layers = [] # e.g. [self.vision_encoder.features[-1]]
            cam = GradCAM(model=self.vision_encoder, target_layers=target_layers, use_cuda=(self.device.type=='cuda'))

        # Generate Mask
        grayscale_cam = cam(input_tensor=img_tensor, targets=None) # Automatically finds highest scoring class if targets=None
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert original Image to numpy 0-1 range
        orig_img_np = np.array(orig_img.resize((Config.MAX_IMAGE_SIZE, Config.MAX_IMAGE_SIZE))) / 255.0
        
        # Apply Heatmap
        visualization = show_cam_on_image(orig_img_np, grayscale_cam, use_rgb=True)
        return visualization, grayscale_cam
