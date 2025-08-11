import os
import sys
import glob
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from rich.console import Console
from collections import defaultdict, deque
import math

from config.settings import CONFIG, ACTION_COLORS
from models.detector import PersonDetector
from models.tracker import DeepSORT
from models.action_recognizer import ActionRecognizer

console = Console()

# Supported video formats
INPUT_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
DEFAULT_CONFIDENCE = 0.25
DEFAULT_ACTION_THRESHOLD = 0.3


class ActionSmoother:
    """Smooths action predictions over time to reduce flickering"""
    
    def __init__(self, window_size=5, confidence_threshold=0.3):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.track_histories = defaultdict(lambda: deque(maxlen=window_size))
        
    def smooth_action(self, track_id, action, confidence):
        """Apply temporal smoothing to action predictions"""
        history = self.track_histories[track_id]
        
        # Add current prediction to history
        history.append((action, confidence))
        
        # If we don't have enough history, return current prediction
        if len(history) < min(3, self.window_size):
            return action, confidence
        
        # Count action occurrences with weighted confidence
        action_scores = defaultdict(float)
        total_weight = 0
        
        for i, (hist_action, hist_conf) in enumerate(history):
            # Give more weight to recent predictions
            weight = (i + 1) / len(history)
            if hist_conf >= self.confidence_threshold:
                action_scores[hist_action] += weight * hist_conf
                total_weight += weight
        
        # Find the action with highest weighted score
        if action_scores and total_weight > 0:
            best_action = max(action_scores.items(), key=lambda x: x[1])
            smoothed_action = best_action[0]
            smoothed_confidence = best_action[1] / total_weight
            return smoothed_action, min(smoothed_confidence, 1.0)
        
        return action, confidence


def draw_enhanced_box_label(img, box, track_id, action, confidence, color=(0, 255, 0)):
    """Draw enhanced bounding box with improved styling and labels"""
    x1, y1, x2, y2 = map(int, box)
    
    # Ensure coordinates are valid
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return
    
    # Calculate adaptive line width based on box size and image resolution
    box_area = (x2 - x1) * (y2 - y1)
    img_area = h * w
    relative_size = box_area / img_area
    
    # Adaptive line width (thicker for larger boxes and higher resolution)
    base_lw = max(1, int(math.sqrt(img_area) * 0.001))
    lw = max(1, int(base_lw * (1 + relative_size * 2)))
    
    # Draw main bounding box with rounded corners effect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=lw, lineType=cv2.LINE_AA)
    
    # Draw corner accents for better visibility
    corner_length = min(25, (x2 - x1) // 8, (y2 - y1) // 8)
    accent_thickness = max(2, lw + 1)
    
    # Corner accents (L-shaped markers)
    corners = [
        # Top-left
        [(x1, y1), (x1 + corner_length, y1), (x1, y1 + corner_length)],
        # Top-right  
        [(x2, y1), (x2 - corner_length, y1), (x2, y1 + corner_length)],
        # Bottom-left
        [(x1, y2), (x1 + corner_length, y2), (x1, y2 - corner_length)],
        # Bottom-right
        [(x2, y2), (x2 - corner_length, y2), (x2, y2 - corner_length)]
    ]
    
    for corner in corners:
        cv2.line(img, corner[0], corner[1], color, accent_thickness, cv2.LINE_AA)
        cv2.line(img, corner[0], corner[2], color, accent_thickness, cv2.LINE_AA)
    
    # Enhanced label rendering
    font_scale = max(0.4, min(0.8, relative_size * 3))
    font_thickness = max(1, int(font_scale * 2))
    
    # Create informative label text
    confidence_text = f"{confidence:.1%}" if confidence > 0 else "N/A"
    lines = [
        f"ID: {track_id}",
        f"{action.upper()}",
        f"Conf: {confidence_text}"
    ]
    
    # Calculate text dimensions
    line_heights = []
    line_widths = []
    
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        line_widths.append(w)
        line_heights.append(h + baseline)
    
    max_width = max(line_widths)
    total_height = sum(line_heights)
    
    # Label positioning and styling
    padding = max(4, int(font_scale * 8))
    label_width = max_width + 2 * padding
    label_height = total_height + (len(lines) + 1) * padding
    
    # Position label above box if space available, otherwise below
    if y1 - label_height - 10 > 0:
        label_y = y1 - label_height - 5
        text_start_y = label_y + line_heights[0] + padding
    else:
        label_y = y2 + 5
        text_start_y = label_y + line_heights[0] + padding
    
    # Ensure label stays within image bounds
    label_x = max(0, min(x1, w - label_width))
    label_y = max(0, min(label_y, h - label_height))
    
    # Draw label background with gradient effect
    overlay = img.copy()
    
    # Main background
    cv2.rectangle(overlay, (label_x, label_y), 
                 (label_x + label_width, label_y + label_height), 
                 color, -1)
    
    # Darker accent border
    darker_color = tuple(int(c * 0.7) for c in color)
    cv2.rectangle(overlay, (label_x, label_y), 
                 (label_x + label_width, label_y + label_height), 
                 darker_color, 2)
    
    # Apply transparency
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text with shadow effect
    text_color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    
    current_y = text_start_y
    for i, line in enumerate(lines):
        text_x = label_x + padding
        
        # Draw text shadow
        cv2.putText(img, line, (text_x + 1, current_y + 1), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, 
                   font_thickness, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(img, line, (text_x, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
                   font_thickness, cv2.LINE_AA)
        
        current_y += line_heights[i] + padding
    
    # Add confidence indicator bar
    if confidence > 0:
        bar_width = max_width
        bar_height = 4
        bar_x = label_x + padding
        bar_y = label_y + label_height - bar_height - padding
        
        # Background bar
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 0, 255)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                     conf_color, -1)


def process_video(input_path: Path, output_path: Path, detector, tracker, action_recognizer):
    """Process a single video file with enhanced tracking and action recognition"""
    
    try:
        # Initialize action smoother
        action_smoother = ActionSmoother(window_size=7, confidence_threshold=0.3)
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            console.print(f"[red]Error: Cannot open video {input_path}[/red]")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            console.print(f"[red]Error: Cannot create output video {output_path}[/red]")
            cap.release()
            return False
        
        # Process frames
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        print(f"Processing {input_path.name}...")
        
        # Suppress detector output
        import io
        import contextlib
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Person detection
            with contextlib.redirect_stdout(io.StringIO()):
                detections = detector.detect_persons(frame)
                tracks = tracker.update(detections, frame)
            
            # Update detection count
            detection_count += len(tracks)
            
            # Process each track
            for track in tracks:
                if len(track) >= 5:  # Ensure we have x1, y1, x2, y2, track_id
                    x1, y1, x2, y2, track_id = track[:5]
                    
                    # Ensure coordinates are within bounds
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(width, int(x2)), min(height, int(y2))
                    
                    if x2 > x1 and y2 > y1:
                        # Extract crop for action recognition
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            with contextlib.redirect_stdout(io.StringIO()):
                                # Get action prediction
                                action, confidence = action_recognizer.recognize_action_with_tracking(track_id, crop)
                                
                                # Apply smoothing to reduce flickering
                                action, confidence = action_smoother.smooth_action(track_id, action, confidence)
                            
                            # Get color for action (with fallback)
                            color = ACTION_COLORS.get(action, ACTION_COLORS.get('unknown', (0, 255, 0)))
                            
                            # Draw enhanced bounding box and label
                            draw_enhanced_box_label(frame, (x1, y1, x2, y2), track_id, action, confidence, color)
            
            # Add frame information overlay (bottom-left)
            info_overlay = frame.copy()
            info_text = f"Frame: {frame_count + 1}/{total_frames} | Active Tracks: {len(tracks)} | Total Detections: {detection_count}"
            
            # Calculate text size for background
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Draw semi-transparent background
            cv2.rectangle(info_overlay, (5, height - text_height - 15), 
                         (text_width + 15, height - 5), (0, 0, 0), -1)
            cv2.addWeighted(info_overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw text
            cv2.putText(frame, info_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Progress update
            elapsed_time = time.time() - start_time
            progress_percent = (frame_count / total_frames) * 100
            fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
            eta = ((total_frames - frame_count) / fps_current) if fps_current > 0 else 0
            
            print(f"\rFrame {frame_count:4d}/{total_frames} ({progress_percent:5.1f}%) | "
                  f"Active: {len(tracks):3d} | Total: {detection_count:4d} | "
                  f"Speed: {fps_current:.1f} fps | ETA: {eta:.1f}s" + " " * 10, 
                  end="", flush=True)
        
        print()  # Final newline
        
        # Cleanup
        cap.release()
        out.release()
        
        console.print(f"[green]✓ Processed {input_path.name} -> {output_path.name} ({detection_count} total detections)[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Error processing {input_path.name}: {str(e)}[/red]")
        return False


def main():
    """Main function for enhanced batch processing"""
    parser = argparse.ArgumentParser(description="Enhanced Batch Video Action Recognition Processor")
    parser.add_argument("--input-dir", required=True, help="Input directory containing videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed videos")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, 
                       help="Detection confidence threshold")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_ACTION_THRESHOLD,
                       help="Action recognition threshold")
    parser.add_argument("--skip-existing", action="store_true", 
                       help="Skip processing if output file already exists")
    parser.add_argument("--max-area", type=int, default=1200000,
                       help="Maximum detection area (pixels)")
    parser.add_argument("--min-area", type=int, default=1000,
                       help="Minimum detection area (pixels)")
    parser.add_argument("--smoothing-window", type=int, default=7,
                       help="Window size for action smoothing")
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory {input_dir} does not exist[/red]")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = []
    for ext in INPUT_EXTENSIONS:
        video_files.extend(list(input_dir.glob(ext)))
    
    if not video_files:
        console.print(f"[red]No video files found in {input_dir}[/red]")
        sys.exit(1)
    
    # Sort files for consistent processing order
    video_files = sorted(video_files)
    
    console.print(f"[cyan]Processing {len(video_files)} videos from {input_dir} to {output_dir}[/cyan]")
    
    # Initialize models
    console.print("[yellow]Initializing models...[/yellow]")
    start_time = time.time()
    
    try:
        detector = PersonDetector(CONFIG["yolo_model"], args.device, args.confidence)
        
        # Update detector settings if available
        if hasattr(detector, 'max_area'):
            detector.max_area = args.max_area
        if hasattr(detector, 'min_area'):
            detector.min_area = args.min_area
        if hasattr(detector, 'verbose'):
            detector.verbose = False
            
        tracker = DeepSORT(CONFIG["max_age"], CONFIG["min_hits"], CONFIG["iou_threshold_track"])
        action_recognizer = ActionRecognizer(args.device, CONFIG["sequence_length"], args.action_threshold)
        
        # Configure action recognizer for better accuracy
        if hasattr(action_recognizer, 'set_smoothing'):
            action_recognizer.set_smoothing(True)
        
        init_time = time.time() - start_time
        console.print(f"[green]✓ Models initialized in {init_time:.2f}s[/green]")
        
    except Exception as e:
        console.print(f"[red]Error initializing models: {str(e)}[/red]")
        sys.exit(1)
    
    # Process each video
    successful = 0
    failed = 0
    skipped = 0
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        console.print(f"\n[cyan]Processing {i}/{len(video_files)}: {video_path.name}[/cyan]")
        
        # Determine output path
        output_path = output_dir / f"processed_{video_path.stem}.mp4"
        
        # Skip if output exists and skip_existing is enabled
        if args.skip_existing and output_path.exists():
            console.print(f"[yellow]Skipping {video_path.name} (already exists)[/yellow]")
            skipped += 1
            continue
        
        # Process video
        video_start_time = time.time()
        success = process_video(video_path, output_path, detector, tracker, action_recognizer)
        video_time = time.time() - video_start_time
        
        if success:
            successful += 1
            console.print(f"[green]Completed in {video_time:.2f}s[/green]")
        else:
            failed += 1
            console.print(f"[red]Failed after {video_time:.2f}s[/red]")
    
    # Final summary
    total_time = time.time() - total_start_time
    console.print(f"\n[bold]Processing Summary:[/bold]")
    console.print(f"✓ Successful: {successful}")
    console.print(f"✗ Failed: {failed}")
    console.print(f"⏭ Skipped: {skipped}")
    console.print(f"⏱ Total time: {total_time:.2f}s")
    if len(video_files) > 0:
        console.print(f"⚡ Average per video: {total_time/len(video_files):.2f}s")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()