import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional


class PersonDetector:
    """
    Person detection using YOLO model.
    Detects persons in images/video frames and extracts features for tracking.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", confidence: float = 0.3):
        """
        Initialize person detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('cpu' or 'cuda')
            confidence: Confidence threshold for detections
        """
        self.device = device
        self.confidence = confidence
        self.person_class_id = 0  # COCO class ID for person
        
        # Load YOLO model
        try:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            if device == "cuda" and torch.cuda.is_available():
                self.model.to(device)
                print("Using CUDA device")
            else:
                self.model.to("cpu")
                self.device = "cpu"
                print("Using CPU device")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")
        
        # Detection filters - Made more lenient
        self.min_area = 100  # Reduced from 200
        self.max_area = 500000  # Increased from 300000
        self.min_aspect_ratio = 0.1  # Reduced from 0.2
        self.max_aspect_ratio = 10.0  # Increased from 5.0
        
    def detect_persons(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detections as (x1, y1, x2, y2, confidence)
        """
        if image is None or image.size == 0:
            return []
        
        try:
            # Run YOLO inference with verbose output for debugging
            results = self.model(image, conf=self.confidence, verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"Found {len(result.boxes)} total detections")
                    
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        print(f"Detection {i}: Class {class_id}, Confidence {conf:.3f}")
                        
                        # Check if detection is a person
                        if class_id != self.person_class_id:
                            continue
                        
                        # Extract coordinates and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        print(f"Person detection: ({x1}, {y1}, {x2}, {y2}), conf={conf:.3f}")
                        
                        # Apply filters with debug info
                        if self._is_valid_detection(x1, y1, x2, y2, image.shape):
                            detections.append((x1, y1, x2, y2, conf))
                            print(f"Valid detection added: ({x1}, {y1}, {x2}, {y2})")
                        else:
                            print(f"Detection filtered out: ({x1}, {y1}, {x2}, {y2})")
                else:
                    print("No boxes found in YOLO results")
            else:
                print("No results from YOLO model")
            
            print(f"Total valid person detections: {len(detections)}")
            return detections
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_persons_with_features(self, image: np.ndarray) -> Dict:
        """
        Detect persons and extract features for tracking.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with 'detections' and 'features' keys
        """
        detections = self.detect_persons(image)
        
        # Extract features from detection crops
        features = []
        for x1, y1, x2, y2, conf in detections:
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            # Extract simple features (can be replaced with deep features)
            feature = self._extract_features(crop)
            features.append(feature)
        
        return {
            "detections": detections,
            "features": features
        }
    
    def _is_valid_detection(self, x1: int, y1: int, x2: int, y2: int, 
                           image_shape: Tuple[int, int, int]) -> bool:
        """
        Validate detection based on size and aspect ratio constraints.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            True if detection is valid, False otherwise
        """
        # Check if coordinates are valid
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return False
        
        # Check if detection is within image bounds
        height, width = image_shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"Detection outside image bounds: ({x1}, {y1}, {x2}, {y2}), image: {width}x{height}")
            return False
        
        # Calculate area and aspect ratio
        area = (x2 - x1) * (y2 - y1)
        aspect_ratio = (y2 - y1) / max(x2 - x1, 1)
        
        print(f"Detection area: {area}, aspect_ratio: {aspect_ratio:.2f}")
        
        # Apply area constraints
        if area < self.min_area:
            print(f"Detection too small: {area} < {self.min_area}")
            return False
        if area > self.max_area:
            print(f"Detection too large: {area} > {self.max_area}")
            return False
        
        # Apply aspect ratio constraints
        if aspect_ratio < self.min_aspect_ratio:
            print(f"Aspect ratio too small: {aspect_ratio} < {self.min_aspect_ratio}")
            return False
        if aspect_ratio > self.max_aspect_ratio:
            print(f"Aspect ratio too large: {aspect_ratio} > {self.max_aspect_ratio}")
            return False
        
        return True
    
    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract features from person crop for tracking.
        
        Args:
            crop: Person crop as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        if crop.size == 0:
            return np.zeros(128)  # Return zero feature vector
        
        try:
            # Resize crop to standard size
            resized = cv2.resize(crop, (64, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Extract histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalize
            hist = hist / (np.sum(hist) + 1e-6)
            
            # Pad or truncate to fixed size
            if len(hist) < 128:
                hist = np.pad(hist, (0, 128 - len(hist)), 'constant')
            else:
                hist = hist[:128]
            
            return hist.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(128)
    
    def test_detection_on_frame(self, image: np.ndarray) -> None:
        """
        Test detection on a single frame with detailed output.
        """
        print(f"Testing detection on frame of size: {image.shape}")
        
        # Try with very low confidence first
        original_conf = self.confidence
        self.confidence = 0.01
        
        try:
            results = self.model(image, conf=self.confidence, verbose=True)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    print(f"Raw detections: {len(result.boxes)}")
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        print(f"  {i}: Class {class_id}, Conf {conf:.3f}, Box ({x1}, {y1}, {x2}, {y2})")
                else:
                    print("No boxes in results")
            else:
                print("No results from model")
                
        finally:
            self.confidence = original_conf
    
    def update_detection_params(self, min_area: Optional[int] = None,
                               max_area: Optional[int] = None,
                               min_aspect_ratio: Optional[float] = None,
                               max_aspect_ratio: Optional[float] = None,
                               confidence: Optional[float] = None):
        """
        Update detection parameters.
        
        Args:
            min_area: Minimum detection area
            max_area: Maximum detection area
            min_aspect_ratio: Minimum aspect ratio (height/width)
            max_aspect_ratio: Maximum aspect ratio (height/width)
            confidence: Confidence threshold
        """
        if min_area is not None:
            self.min_area = min_area
        if max_area is not None:
            self.max_area = max_area
        if min_aspect_ratio is not None:
            self.min_aspect_ratio = min_aspect_ratio
        if max_aspect_ratio is not None:
            self.max_aspect_ratio = max_aspect_ratio
        if confidence is not None:
            self.confidence = confidence
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[int, int, int, int, float]], 
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detections as (x1, y1, x2, y2, confidence)
            color: Color for bounding boxes (BGR format)
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for x1, y1, x2, y2, conf in detections:
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            label = f"Person: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image
    
    def get_detection_stats(self, detections: List[Tuple[int, int, int, int, float]]) -> Dict:
        """
        Get statistics about detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "avg_area": 0.0,
                "avg_aspect_ratio": 0.0
            }
        
        confidences = [conf for _, _, _, _, conf in detections]
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _ in detections]
        aspect_ratios = [(y2 - y1) / max(x2 - x1, 1) for x1, y1, x2, y2, _ in detections]
        
        return {
            "count": len(detections),
            "avg_confidence": np.mean(confidences),
            "avg_area": np.mean(areas),
            "avg_aspect_ratio": np.mean(aspect_ratios),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "min_area": np.min(areas),
            "max_area": np.max(areas)
        }


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = PersonDetector(model_path="yolov8n.pt", device="cpu", confidence=0.3)
    
    # Load and process image
    image = cv2.imread("example.jpg")
    if image is not None:
        # Test detection first
        detector.test_detection_on_frame(image)
        
        # Detect persons
        detections = detector.detect_persons(image)
        print(f"Found {len(detections)} persons")
        
        # Get detection statistics
        stats = detector.get_detection_stats(detections)
        print(f"Detection stats: {stats}")
        
        # Draw detections
        result_image = detector.draw_detections(image, detections)
        cv2.imshow("Person Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Detect with features for tracking
        result_with_features = detector.detect_persons_with_features(image)
        print(f"Extracted {len(result_with_features['features'])} feature vectors")
    else:
        print("Could not load image")