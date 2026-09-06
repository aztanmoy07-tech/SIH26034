import cv2
import numpy as np

class QualityGateResult:
    def __init__(self, passed: bool, reason: str = "", blur_score: float = 0.0, glare_score: float = 0.0):
        self.passed = passed
        self.reason = reason
        self.blur_score = blur_score
        self.glare_score = glare_score
        
    def to_dict(self):
        return {
            "passed": self.passed,
            "reason": self.reason,
            "blur_score": round(self.blur_score, 2),
            "glare_score": round(self.glare_score, 2)
        }

class ImageQualityAssessor:
    """Phase 3: Image quality gating before OCR analysis."""
    
    # Thresholds
    BLUR_THRESHOLD = 50.0   # Variance of Laplacian
    GLARE_THRESHOLD = 0.05  # Max 5% of pixels can be blown out (254-255)
    MIN_RESOLUTION = (800, 600)
    
    @classmethod
    def evaluate(cls, img_np: np.ndarray) -> QualityGateResult:
        if img_np is None or img_np.size == 0:
            return QualityGateResult(False, "Invalid image data.")
            
        h, w = img_np.shape[:2]
        
        # 1. Resolution Check
        if h < cls.MIN_RESOLUTION[1] or w < cls.MIN_RESOLUTION[0]:
            return QualityGateResult(False, f"Image resolution too low ({w}x{h}). Move closer or use a higher quality camera.")
            
        # Convert to grayscale for blur and glare checks
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        
        # 2. Blur Check (Variance of Laplacian)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < cls.BLUR_THRESHOLD:
            return QualityGateResult(False, "Image is too blurry. Hold the phone steady and refocus.", blur_score=blur_score)
            
        # 3. Glare Check
        # Count pixels near max brightness
        glare_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
        glare_ratio = cv2.countNonZero(glare_mask) / (h * w)
        if glare_ratio > cls.GLARE_THRESHOLD:
            return QualityGateResult(
                False, 
                "Severe glare detected on packaging. Change the angle to reduce reflections.",
                blur_score=blur_score,
                glare_score=glare_ratio
            )
            
        return QualityGateResult(True, "Quality checks passed.", blur_score=blur_score, glare_score=glare_ratio)
