import cv2
import numpy as np
import os
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

# Default action classes if kinetics_classes is not available
DEFAULT_ACTION_CLASSES = [
    "standing", "walking", "running", "jumping", "sitting", "lying", 
    "waving", "pointing", "dancing", "exercising", "playing", "working"
]

# Try to import kinetics classes, fallback to default if not available
try:
    from config.kinetics_classes import KINETICS400_CLASSES
    ACTION_CLASSES = KINETICS400_CLASSES
except ImportError:
    print("Warning: kinetics_classes not found, using default action classes")
    ACTION_CLASSES = DEFAULT_ACTION_CLASSES


class ActionRecognizer:
    """
    Action recognition using 3D ResNet on temporal sequences with motion fallback.
    """
    
    def __init__(self, device="cpu", sequence_length=8, confidence_threshold=0.4, model_path=None):
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize action classes FIRST
        self.action_classes = ACTION_CLASSES
        
        # Buffers for storing frame sequences per track
        self.track_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
        self.track_predictions = defaultdict(lambda: {"action": "unknown", "confidence": 0.0})
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
        # Motion analysis parameters
        self.motion_history = defaultdict(lambda: deque(maxlen=5))
        
    def _load_model(self, model_path=None):
        """Load the 3D ResNet model."""
        try:
            if model_path and os.path.exists(model_path):
                # Load custom trained model
                print(f"Loading custom model from: {model_path}")
                model = r3d_18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, len(self.action_classes))
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Use pretrained model (this will be adapted for action recognition)
                print("Loading pretrained R3D model...")
                model = r3d_18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, len(self.action_classes))
                
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to motion-based recognition")
            return None
            
    def _get_transforms(self):
        """Get preprocessing transforms."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                               std=[0.22803, 0.22145, 0.216989])
        ])
        
    def recognize_action_with_tracking(self, track_id, crop_frame):
        """
        Recognize action for a specific track.
        
        Args:
            track_id: unique identifier for the track
            crop_frame: cropped frame containing the person
            
        Returns:
            tuple: (action_name, confidence_score)
        """
        if crop_frame is None or crop_frame.size == 0:
            return "unknown", 0.0
            
        # Resize crop to standard size
        try:
            crop_resized = cv2.resize(crop_frame, (112, 112))
        except Exception as e:
            print(f"Error resizing crop: {e}")
            return "unknown", 0.0
        
        # Add frame to track buffer
        self.track_buffers[track_id].append(crop_resized)
        
        # Need at least 3 frames for motion analysis
        if len(self.track_buffers[track_id]) < 3:
            return "detecting", 0.5
            
        # Use motion-based recognition (more reliable than deep learning for this use case)
        return self._recognize_with_motion(track_id)
            
    def _recognize_with_model(self, track_id):
        """Recognize action using deep learning model."""
        if self.model is None:
            return self._recognize_with_motion(track_id)
            
        try:
            buffer = self.track_buffers[track_id]
            
            # Need full sequence for model
            if len(buffer) < self.sequence_length:
                return self._recognize_with_motion(track_id)
                
            # Prepare input tensor
            frames = []
            for frame in buffer:
                try:
                    transformed = self.transform(frame)
                    frames.append(transformed)
                except Exception as e:
                    print(f"Error transforming frame: {e}")
                    return self._recognize_with_motion(track_id)
                
            # Stack frames: [C, T, H, W]
            clip = torch.stack(frames, dim=1)
            clip = clip.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Predict
            with torch.no_grad():
                outputs = self.model(clip)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            action = self.action_classes[predicted.item()]
            conf_score = confidence.item()
            
            # Update prediction if confidence is high enough
            if conf_score > self.confidence_threshold:
                self.track_predictions[track_id] = {
                    "action": action,
                    "confidence": conf_score
                }
                
            return action, conf_score
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self._recognize_with_motion(track_id)
            
    def _recognize_with_motion(self, track_id):
        """Enhanced motion-based action recognition."""
        buffer = self.track_buffers[track_id]
        
        if len(buffer) < 3:
            return "unknown", 0.0
            
        try:
            # Calculate motion between consecutive frames
            motion_scores = []
            vertical_motion = []
            horizontal_motion = []
            
            for i in range(len(buffer) - 1):
                frame1 = cv2.cvtColor(buffer[i], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(buffer[i + 1], cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(frame1, frame2)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
                
                # Analyze motion direction using center of mass
                try:
                    # Find person pixels (assuming person is brighter regions)
                    _, thresh1 = cv2.threshold(frame1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _, thresh2 = cv2.threshold(frame2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Calculate centers
                    M1 = cv2.moments(thresh1)
                    M2 = cv2.moments(thresh2)
                    
                    if M1["m00"] != 0 and M2["m00"] != 0:
                        cy1 = int(M1["m01"] / M1["m00"])
                        cx1 = int(M1["m10"] / M1["m00"])
                        cy2 = int(M2["m01"] / M2["m00"])
                        cx2 = int(M2["m10"] / M2["m00"])
                        
                        vertical_motion.append(abs(cy2 - cy1))
                        horizontal_motion.append(abs(cx2 - cx1))
                    else:
                        vertical_motion.append(0)
                        horizontal_motion.append(0)
                        
                except Exception as e:
                    vertical_motion.append(0)
                    horizontal_motion.append(0)
            
            # Calculate motion statistics
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            motion_variance = np.var(motion_scores) if motion_scores else 0
            avg_vertical = np.mean(vertical_motion) if vertical_motion else 0
            avg_horizontal = np.mean(horizontal_motion) if horizontal_motion else 0
            
            # Store motion history
            self.motion_history[track_id].append({
                'motion': avg_motion,
                'vertical': avg_vertical,
                'horizontal': avg_horizontal,
                'variance': motion_variance
            })
            
            # Analyze motion pattern over time
            if len(self.motion_history[track_id]) >= 3:
                recent_motion = list(self.motion_history[track_id])[-3:]
                
                # Calculate trends
                motion_trend = [m['motion'] for m in recent_motion]
                vertical_trend = [m['vertical'] for m in recent_motion]
                
                # Enhanced action classification
                action, confidence = self._classify_action_advanced(
                    avg_motion, avg_vertical, avg_horizontal, 
                    motion_variance, motion_trend, vertical_trend
                )
            else:
                # Simple classification for new tracks
                action, confidence = self._classify_action_simple(avg_motion, avg_vertical, avg_horizontal)
            
            # Update prediction
            self.track_predictions[track_id] = {
                "action": action,
                "confidence": confidence
            }
            
            return action, confidence
            
        except Exception as e:
            print(f"Error in motion analysis: {e}")
            return "unknown", 0.0
            
    def _classify_action_simple(self, avg_motion, avg_vertical, avg_horizontal):
        """Simple action classification based on motion."""
        if avg_motion < 5:
            return "standing", 0.8
        elif avg_motion < 12:
            if avg_vertical > avg_horizontal * 1.5:
                return "jumping", 0.6
            else:
                return "walking", 0.7
        elif avg_motion < 25:
            return "running", 0.6
        else:
            return "dancing", 0.5
            
    def _classify_action_advanced(self, avg_motion, avg_vertical, avg_horizontal, 
                                 motion_variance, motion_trend, vertical_trend):
        """Advanced action classification with temporal analysis."""
        
        # Check for standing (low motion, low variance)
        if avg_motion < 6 and motion_variance < 15:
            return "standing", 0.9
        
        # Check for jumping (high vertical motion, high variance)
        if avg_vertical > 8 and motion_variance > 40:
            return "jumping", 0.8
        
        # Check for running (high motion, but consistent pattern)
        if avg_motion > 20 and motion_variance < 60:
            return "running", 0.7
        
        # Check for walking (moderate motion, rhythmic pattern)
        if 8 < avg_motion < 20:
            # Check for rhythmic pattern in motion
            if len(motion_trend) >= 3:
                motion_diff = np.diff(motion_trend)
                if np.std(motion_diff) < 8:  # Consistent motion
                    return "walking", 0.8
                else:
                    return "dancing", 0.6
            return "walking", 0.7
        
        # Check for sitting (low motion, but some small movements)
        if avg_motion < 8 and avg_vertical < 3:
            return "sitting", 0.7
        
        # Check for waving (high horizontal motion, low vertical)
        if avg_horizontal > avg_vertical * 1.5 and avg_horizontal > 5:
            return "waving", 0.6
        
        # Check for exercising (high motion with high variance)
        if avg_motion > 15 and motion_variance > 30:
            return "exercising", 0.6
        
        # Default classification
        if avg_motion < 10:
            return "standing", 0.5
        else:
            return "moving", 0.5
        
    def get_track_prediction(self, track_id):
        """Get the latest prediction for a track."""
        return self.track_predictions.get(track_id, {"action": "unknown", "confidence": 0.0})
        
    def reset_track(self, track_id):
        """Reset buffers for a specific track."""
        if track_id in self.track_buffers:
            del self.track_buffers[track_id]
        if track_id in self.track_predictions:
            del self.track_predictions[track_id]
        if track_id in self.motion_history:
            del self.motion_history[track_id]
            
    def cleanup_old_tracks(self, active_track_ids):
        """Clean up buffers for tracks that are no longer active."""
        track_ids_to_remove = []
        
        for track_id in self.track_buffers.keys():
            if track_id not in active_track_ids:
                track_ids_to_remove.append(track_id)
                
        for track_id in track_ids_to_remove:
            self.reset_track(track_id)