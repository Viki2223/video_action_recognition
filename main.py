import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import platform
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity

# Configuration and color mapping
CONFIG = {
    'detector': {
        'model_path': 'yolov8n.pt',
        'confidence': 0.3,
        'device': 'cpu'
    },
    'tracker': {
        'max_age': 30,
        'min_hits': 3,
        'iou_threshold': 0.3
    },
    'action_recognizer': {
        'sequence_length': 8,
        'confidence_threshold': 0.4,
        'device': 'cpu'
    },
    'output': {
        'save_video': True,
        'save_results': True,
        'display_video': False
    }
}

ACTION_COLORS = {
    'standing': (0, 255, 0),      # Green
    'walking': (255, 0, 0),       # Blue
    'running': (0, 0, 255),       # Red
    'jumping': (255, 255, 0),     # Cyan
    'sitting': (255, 0, 255),     # Magenta
    'lying': (0, 255, 255),       # Yellow
    'waving': (128, 0, 128),      # Purple
    'dancing': (255, 128, 0),     # Orange
    'exercising': (0, 128, 255),  # Light Blue
    'moving': (128, 128, 128),    # Gray
    'unknown': (64, 64, 64),      # Dark Gray
    'detecting': (255, 255, 255)  # White
}

def convert_windows_path_to_wsl(path_str):
    """Convert Windows path to WSL path if running in WSL"""
    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        # Running in WSL
        if path_str.startswith("C:") or path_str.startswith("c:"):
            # Convert C:\ to /mnt/c/
            wsl_path = path_str.replace("C:", "/mnt/c").replace("c:", "/mnt/c")
            wsl_path = wsl_path.replace("\\", "/")
            return wsl_path
        elif path_str.startswith("D:") or path_str.startswith("d:"):
            # Convert D:\ to /mnt/d/
            wsl_path = path_str.replace("D:", "/mnt/d").replace("d:", "/mnt/d")
            wsl_path = wsl_path.replace("\\", "/")
            return wsl_path
        elif path_str.startswith("E:") or path_str.startswith("e:"):
            # Convert E:\ to /mnt/e/
            wsl_path = path_str.replace("E:", "/mnt/e").replace("e:", "/mnt/e")
            wsl_path = wsl_path.replace("\\", "/")
            return wsl_path
        # Add more drive letters as needed
        elif ":" in path_str:
            # Generic drive letter conversion
            drive_letter = path_str[0].lower()
            wsl_path = path_str.replace(f"{drive_letter}:", f"/mnt/{drive_letter}")
            wsl_path = wsl_path.replace("\\", "/")
            return wsl_path
    return path_str

def check_display_support():
    """Check if display functionality is available"""
    try:
        # Try to create a dummy window to test display support
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False

class PersonDetector:
    """Person detection using YOLO"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu', confidence: float = 0.3):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        
    def detect_persons(self, frame: np.ndarray) -> List[Tuple]:
        """Detect persons in frame and return bounding boxes"""
        results = self.model(frame, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a person (class 0 in COCO)
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return detections

class DeepSORT:
    """Simplified tracking system"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List[Tuple]) -> np.ndarray:
        """Update tracks with new detections"""
        # Simple tracking: match detections to existing tracks by IoU
        updated_tracks = []
        used_detections = set()
        
        # Update existing tracks
        for track_id, track_info in list(self.tracks.items()):
            best_match = None
            best_iou = 0
            best_detection_idx = -1
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                    
                iou = self.calculate_iou(track_info['bbox'], detection)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = detection
                    best_detection_idx = i
            
            if best_match is not None:
                # Update track
                self.tracks[track_id]['bbox'] = best_match[:4]
                self.tracks[track_id]['confidence'] = best_match[4]
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
                used_detections.add(best_detection_idx)
                
                if self.tracks[track_id]['hits'] >= self.min_hits:
                    updated_tracks.append(list(best_match[:4]) + [track_id])
            else:
                # Age the track
                self.tracks[track_id]['age'] += 1
                
                # Remove old tracks
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
                elif self.tracks[track_id]['hits'] >= self.min_hits:
                    # Keep track even if not matched this frame
                    bbox = self.tracks[track_id]['bbox']
                    updated_tracks.append(list(bbox) + [track_id])
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self.tracks[self.next_id] = {
                    'bbox': detection[:4],
                    'confidence': detection[4],
                    'age': 0,
                    'hits': 1
                }
                self.next_id += 1
        
        return np.array(updated_tracks) if updated_tracks else np.array([]).reshape(0, 5)

class ActionRecognizer:
    """Action recognition system"""
    
    def __init__(self, device: str = 'cpu', sequence_length: int = 8, confidence_threshold: float = 0.4):
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.track_sequences = defaultdict(deque)
        
        # Simple action mapping based on movement patterns
        self.actions = ['standing', 'walking', 'running', 'jumping', 'sitting', 'lying', 'waving', 'unknown']
        
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract simple features from person crop"""
        if crop is None or crop.size == 0:
            return np.zeros(10)
        
        # Simple features: aspect ratio, area, color histograms
        height, width = crop.shape[:2]
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        # Color histogram features
        if len(crop.shape) == 3:
            hist_b = cv2.calcHist([crop], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([crop], [1], None, [8], [0, 256])
            hist_r = cv2.calcHist([crop], [2], None, [8], [0, 256])
            hist_features = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        else:
            hist_features = np.zeros(24)
        
        # Normalize features
        features = np.array([aspect_ratio, area / 10000] + hist_features[:8].tolist())
        return features
    
    def classify_action(self, features_sequence: List[np.ndarray]) -> Tuple[str, float]:
        """Classify action based on feature sequence"""
        if len(features_sequence) < 2:
            return 'unknown', 0.5
        
        # Simple rule-based classification
        current_features = features_sequence[-1]
        prev_features = features_sequence[-2]
        
        # Calculate movement (simple difference)
        movement = np.linalg.norm(current_features - prev_features)
        aspect_ratio = current_features[0]
        
        # Simple heuristics
        if aspect_ratio > 1.5:  # Wide aspect ratio
            return 'lying', 0.7
        elif aspect_ratio < 0.5:  # Tall aspect ratio
            if movement > 0.1:
                return 'jumping', 0.6
            else:
                return 'standing', 0.8
        else:  # Normal aspect ratio
            if movement > 0.2:
                return 'running', 0.7
            elif movement > 0.1:
                return 'walking', 0.6
            else:
                return 'sitting', 0.5
    
    def recognize_action_with_tracking(self, track_id: int, crop: np.ndarray) -> Tuple[str, float]:
        """Recognize action for a specific track"""
        features = self.extract_features(crop)
        
        # Add to sequence
        self.track_sequences[track_id].append(features)
        
        # Maintain sequence length
        if len(self.track_sequences[track_id]) > self.sequence_length:
            self.track_sequences[track_id].popleft()
        
        # Classify action
        if len(self.track_sequences[track_id]) >= 2:
            action, confidence = self.classify_action(list(self.track_sequences[track_id]))
            return action, confidence
        else:
            return 'detecting', 0.5
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove sequences for inactive tracks"""
        to_remove = []
        for track_id in self.track_sequences:
            if track_id not in active_track_ids:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_sequences[track_id]

class VideoUtils:
    """Utility class for video processing"""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def create_video_writer(output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
        """Create video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

class VideoActionRecognizer:
    """Main video action recognition system"""
    
    def __init__(self, config: Dict = None):
        """Initialize the video action recognition system"""
        self.config = config or CONFIG
        self.display_available = False
        
        # Check display support only if display is requested
        if self.config['output']['display_video']:
            self.display_available = check_display_support()
            if not self.display_available:
                print("Warning: Display functionality not available. Running in headless mode.")
                self.config['output']['display_video'] = False
        
        # Initialize components
        print("Initializing person detector...")
        self.detector = PersonDetector(
            model_path=self.config['detector']['model_path'],
            device=self.config['detector']['device'],
            confidence=self.config['detector']['confidence']
        )
        
        print("Initializing tracker...")
        self.tracker = DeepSORT(
            max_age=self.config['tracker']['max_age'],
            min_hits=self.config['tracker']['min_hits'],
            iou_threshold=self.config['tracker']['iou_threshold']
        )
        
        print("Initializing action recognizer...")
        self.action_recognizer = ActionRecognizer(
            device=self.config['action_recognizer']['device'],
            sequence_length=self.config['action_recognizer']['sequence_length'],
            confidence_threshold=self.config['action_recognizer']['confidence_threshold']
        )
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections_per_frame': [],
            'processing_times': [],
            'track_counts': []
        }
        
    def process_video(self, input_path: str, output_dir: str = None) -> Dict:
        """Process a video file for action recognition"""
        
        # Convert path if needed
        input_path = convert_windows_path_to_wsl(input_path)
        
        # Verify input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        else:
            output_dir = convert_windows_path_to_wsl(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video info
        print(f"Processing video: {input_path}")
        video_info = VideoUtils.get_video_info(input_path)
        print(f"Video info: {video_info}")
        
        # Setup output paths
        input_name = Path(input_path).stem
        output_video_path = os.path.join(output_dir, f"{input_name}_processed.mp4")
        output_results_path = os.path.join(output_dir, f"{input_name}_results.json")
        
        # Initialize video capture and writer
        cap = cv2.VideoCapture(input_path)
        video_writer = None
        
        if self.config['output']['save_video']:
            video_writer = VideoUtils.create_video_writer(
                output_video_path, 
                video_info['width'], 
                video_info['height'], 
                video_info['fps']
            )
        
        # Process frames
        frame_results = []
        frame_idx = 0
        window_created = False
        
        with tqdm(total=video_info['frame_count'], desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Process frame
                result = self.process_frame(frame, frame_idx)
                frame_results.append(result)
                
                # Draw results on frame
                processed_frame = self.draw_results(frame, result)
                
                # Save frame if needed
                if video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Display frame if needed and available
                if self.config['output']['display_video'] and self.display_available:
                    try:
                        cv2.imshow('Action Recognition', processed_frame)
                        window_created = True
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error as e:
                        print(f"Warning: Display error: {e}")
                        self.config['output']['display_video'] = False
                        self.display_available = False
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                self.stats['processed_frames'] += 1
                
                frame_idx += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        
        # Only call destroyAllWindows if a window was actually created
        if window_created and self.display_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore errors when cleaning up windows
        
        # Save results
        if self.config['output']['save_results']:
            self.save_results(frame_results, output_results_path, video_info)
        
        # Generate summary
        summary = self.generate_summary(frame_results, video_info)
        
        return {
            'output_video': output_video_path,
            'results_file': output_results_path,
            'summary': summary,
            'stats': self.stats
        }
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Process a single frame"""
        
        # Detect persons
        detections = self.detector.detect_persons(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Recognize actions for each track
        actions = {}
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            
            # Extract person crop
            crop = frame[y1:y2, x1:x2] if y1 < y2 and x1 < x2 else None
            
            # Recognize action
            if crop is not None and crop.size > 0:
                action, confidence = self.action_recognizer.recognize_action_with_tracking(
                    track_id, crop
                )
                actions[track_id] = {
                    'action': action,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                }
        
        # Clean up old tracks
        active_track_ids = [int(track[4]) for track in tracks]
        self.action_recognizer.cleanup_old_tracks(active_track_ids)
        
        # Update statistics
        self.stats['detections_per_frame'].append(len(detections))
        self.stats['track_counts'].append(len(tracks))
        
        return {
            'frame_idx': frame_idx,
            'detections': detections,
            'tracks': tracks.tolist() if len(tracks) > 0 else [],
            'actions': actions
        }
    
    def draw_results(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detection and action results on frame"""
        frame_copy = frame.copy()
        
        # Draw tracks and actions
        for track_id, action_info in result['actions'].items():
            x1, y1, x2, y2 = action_info['bbox']
            action = action_info['action']
            confidence = action_info['confidence']
            
            # Get color for action
            color = ACTION_COLORS.get(action, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw action label
            label = f"ID:{track_id} {action} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw frame info
        frame_info = f"Frame: {result['frame_idx']}, Tracks: {len(result['actions'])}"
        cv2.putText(frame_copy, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def save_results(self, frame_results: List[Dict], output_path: str, video_info: Dict):
        """Save processing results to JSON file"""
        results = {
            'video_info': video_info,
            'config': self.config,
            'stats': self.stats,
            'frames': frame_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_summary(self, frame_results: List[Dict], video_info: Dict) -> Dict:
        """Generate processing summary"""
        
        # Count actions
        action_counts = {}
        total_detections = 0
        
        for frame_result in frame_results:
            total_detections += len(frame_result['detections'])
            for action_info in frame_result['actions'].values():
                action = action_info['action']
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate statistics
        avg_detections = np.mean(self.stats['detections_per_frame']) if self.stats['detections_per_frame'] else 0
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        avg_tracks = np.mean(self.stats['track_counts']) if self.stats['track_counts'] else 0
        
        return {
            'total_frames': len(frame_results),
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections,
            'avg_tracks_per_frame': avg_tracks,
            'avg_processing_time': avg_processing_time,
            'fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'action_counts': action_counts,
            'most_common_action': max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else 'none'
        }

def main():
    """Main function with simplified output"""
    parser = argparse.ArgumentParser(description='Video Action Recognition System')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output_dir', '-o', help='Output directory for results')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], default='cpu', 
                       help='Device to use for inference')
    parser.add_argument('--confidence', '-conf', type=float, default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--display', action='store_true',
                       help='Display video during processing')
    parser.add_argument('--no_save_video', action='store_true',
                       help='Don\'t save processed video')
    parser.add_argument('--no_save_results', action='store_true',
                       help='Don\'t save results JSON')
    
    args = parser.parse_args()
    
    # Load configuration
    config = CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Update config with command line arguments
    config['detector']['device'] = args.device
    config['action_recognizer']['device'] = args.device
    config['detector']['confidence'] = args.confidence
    config['output']['display_video'] = args.display
    config['output']['save_video'] = not args.no_save_video
    config['output']['save_results'] = not args.no_save_results
    
    # Initialize system
    try:
        system = VideoActionRecognizer(config)
        
        # Process video
        results = system.process_video(args.input_video, args.output_dir)
        
        # Simplified output
        summary = results['summary']
        output_filename = Path(results['output_video']).name
        
        # Single line success message
        print(f"\n✓ Video processed: {summary['total_frames']} frames | Primary action: {summary['most_common_action']} ({summary['action_counts'][summary['most_common_action']]} detections) | Speed: {summary['fps']:.1f} FPS | Output: {output_filename}")
        
        # Optional: Show action breakdown if multiple actions detected
        if len(summary['action_counts']) > 1:
            actions_str = ", ".join([f"{action}: {count}" for action, count in summary['action_counts'].items()])
            print(f"All actions detected: {actions_str}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())