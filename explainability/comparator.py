import numpy as np

class ExplainabilityComparator:
    """
    Compares two different explainability masks (e.g., Grad-CAM vs Integrated Gradients,
    or Model A Grad-CAM vs Model B Grad-CAM).
    """
    @staticmethod
    def calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculates Intersection over Union (IoU) of two attribution maps.
        Args:
            mask_a, mask_b: Heatmap arrays [0, 1] of shape (H, W).
        Returns:
            IoU score (0.0 to 1.0)
        """
        bin_a = (mask_a >= threshold).astype(bool)
        bin_b = (mask_b >= threshold).astype(bool)
        
        intersection = np.logical_and(bin_a, bin_b).sum()
        union = np.logical_or(bin_a, bin_b).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)
        
    @staticmethod
    def calculate_ssim(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """
        Structural Similarity Index (SSIM) between two continuous maps.
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            return ssim(mask_a, mask_b, data_range=1.0)
        except ImportError:
            print("Install scikit-image to calculate SSIM")
            return 0.0

if __name__ == "__main__":
    comparator = ExplainabilityComparator()
    m_a = np.random.rand(224, 224)
    m_b = np.copy(m_a)
    print(f"Test IoU (Same masks): {comparator.calculate_iou(m_a, m_b)}")
