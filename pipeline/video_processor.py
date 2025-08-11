import os
import sys
import glob
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from config.settings import CONFIG, ACTION_COLORS
from models.detector import PersonDetector
from models.tracker import DeepSORT
from models.action_recognizer import ActionRecognizer

console = Console()

# Supported video formats
INPUT_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
DEFAULT_CONFIDENCE = 0.25
DEFAULT_ACTION_THRESHOLD = 0.3


def draw_box_label(img, box, label, color=(0, 255, 0), txt_color=(255, 255, 255)):
    """Draw bounding box with label - Enhanced version"""
    x1, y1, x2, y2 = map(int, box)
    
    # Calculate line width based on image size
    lw = max(round(sum(img.shape[:2]) * 0.003), 2)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=lw, lineType=cv2.LINE_AA)
    
    if label:
        # Get text size
        font_scale = max(0.4, min(img.shape[:2]) / 1000)
        font_thickness = max(1, lw // 2)
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Calculate text position - above box if possible, otherwise below
        if y1 - h - 10 > 0:
            text_y = y1 - 5
            bg_y1 = y1 - h - 10
            bg_y2 = y1
        else:
            text_y = y2 + h + 5
            bg_y1 = y2
            bg_y2 = y2 + h + 10
        
        # Draw label background
        cv2.rectangle(img, (x1, bg_y1), (x1 + w + 10, bg_y2), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, txt_color, thickness=font_thickness, lineType=cv2.LINE_AA)


def process_video(input_path: Path, output_path: Path, device: str, 
                 conf_thresh: float, act_thresh: float):
    """Process a single video file with enhanced tracking display"""
    
    try:
        # Initialize models
        detector = PersonDetector(CONFIG["yolo_model"], device, conf_thresh)
        tracker = DeepSORT(CONFIG["max_age"], CONFIG["min_hits"], CONFIG["iou_threshold_track"])
        action_recognizer = ActionRecognizer(device, CONFIG["sequence_length"], act_thresh)
        
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
        
        # Progress tracking
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Processing {input_path.name}", total=total_frames)
            
            frame_count = 0
            detection_count = 0
            track_count = 0
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect persons
                detections = detector.detect_persons(frame)
                
                # Convert detections to proper format for tracker
                if len(detections) > 0:
                    det_array = np.array(detections)
                    detection_count += len(detections)
                else:
                    det_array = np.empty((0, 5))
                
                # Update tracker
                tracks = tracker.update(det_array, None)
                
                # Process each track
                if len(tracks) > 0:
                    track_count += len(tracks)
                    
                    for track in tracks:
                        # Track format: [x1, y1, x2, y2, track_id]
                        x1, y1, x2, y2, track_id = track
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        track_id = int(track_id)
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(x1 + 1, min(x2, width))
                        y2 = max(y1 + 1, min(y2, height))
                        
                        # Extract crop for action recognition
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            
                            if crop.size > 0:
                                # Recognize action
                                action, confidence = action_recognizer.recognize_action_with_tracking(track_id, crop)
                                
                                # Get color for action
                                color = ACTION_COLORS.get(action, (0, 255, 0))
                                
                                # Create label with track ID, action, and confidence
                                label = f"ID:{track_id} | {action} ({confidence:.2f})"
                                
                                # Draw bounding box with label
                                draw_box_label(frame, (x1, y1, x2, y2), label, color)
                            else:
                                # Fallback for invalid crop
                                color = (128, 128, 128)  # Gray for unknown
                                label = f"ID:{track_id} | unknown (0.00)"
                                draw_box_label(frame, (x1, y1, x2, y2), label, color)
                        else:
                            # Fallback for invalid coordinates
                            color = (128, 128, 128)  # Gray for unknown
                            label = f"ID:{track_id} | unknown (0.00)"
                            draw_box_label(frame, (x1, y1, x2, y2), label, color)
                
                # Add frame counter to video
                cv2.putText(frame, f"Frame: {frame_count + 1}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add detection and track count
                cv2.putText(frame, f"Detections: {detection_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Active Tracks: {len(tracks) if len(tracks) > 0 else 0}", 
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write frame to output video
                out.write(frame)
                
                # Update progress
                progress.update(task, advance=1)
                frame_count += 1
            
            # Cleanup old tracks
            active_track_ids = [int(track[4]) for track in tracks] if len(tracks) > 0 else []
            action_recognizer.cleanup_old_tracks(active_track_ids)
        
        # Release resources
        cap.release()
        out.release()
        
        # Display completion summary
        console.print(f"\n[green]✓ Processing complete![/green]")
        console.print(f"[cyan]Total frames processed:[/cyan] {frame_count}")
        console.print(f"[cyan]Total detections:[/cyan] {detection_count}")
        console.print(f"[cyan]Output saved to:[/cyan] {output_path}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error processing video: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def process_batch(input_dir: Path, output_dir: Path, device: str, 
                 conf_thresh: float, act_thresh: float):
    """Process multiple videos in a directory"""
    
    # Find all video files
    video_files = []
    for ext in INPUT_EXTENSIONS:
        video_files.extend(input_dir.glob(ext))
    
    if not video_files:
        console.print(f"[yellow]No video files found in {input_dir}[/yellow]")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    console.print(f"[cyan]Found {len(video_files)} video files to process[/cyan]")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        console.print(f"\n[bold cyan]Processing: {video_file.name}[/bold cyan]")
        
        # Create output path
        output_file = output_dir / f"processed_{video_file.name}"
        
        # Process video
        if process_video(video_file, output_file, device, conf_thresh, act_thresh):
            successful += 1
        else:
            failed += 1
    
    # Final summary
    console.print(f"\n[bold green]Batch processing complete![/bold green]")
    console.print(f"[green]Successful:[/green] {successful}")
    console.print(f"[red]Failed:[/red] {failed}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Video Action Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python process_video.py --input video.mp4 --output output.mp4
  
  # Process with custom settings
  python process_video.py --input video.mp4 --output output.mp4 --device cuda --confidence 0.3 --action-threshold 0.4
  
  # Batch process directory
  python process_video.py --input-dir ./videos/ --output-dir ./processed/ --device cuda
        """
    )
    
    # Input/Output arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Input video file path")
    group.add_argument("--input-dir", type=str, help="Input directory containing videos")
    
    parser.add_argument("--output", type=str, help="Output video file path (required for single file)")
    parser.add_argument("--output-dir", type=str, help="Output directory (required for batch processing)")
    
    # Model arguments
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", 
                       help="Device to use for inference (default: cpu)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, 
                       help=f"Person detection confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_ACTION_THRESHOLD, 
                       help=f"Action recognition confidence threshold (default: {DEFAULT_ACTION_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")
    
    # Display configuration
    console.print(Panel.fit(
        f"[bold cyan]Video Action Recognition System[/bold cyan]\n"
        f"Device: {args.device}\n"
        f"Detection Confidence: {args.confidence}\n"
        f"Action Threshold: {args.action_threshold}",
        title="Configuration"
    ))
    
    try:
        if args.input:
            # Single video processing
            input_path = Path(args.input)
            output_path = Path(args.output)
            
            if not input_path.exists():
                console.print(f"[red]Error: Input file not found: {input_path}[/red]")
                sys.exit(1)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process video
            if process_video(input_path, output_path, args.device, args.confidence, args.action_threshold):
                console.print(f"[green]✓ Successfully processed video![/green]")
            else:
                console.print(f"[red]✗ Failed to process video![/red]")
                sys.exit(1)
        
        else:
            # Batch processing
            input_dir = Path(args.input_dir)
            output_dir = Path(args.output_dir)
            
            if not input_dir.exists():
                console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
                sys.exit(1)
            
            if not input_dir.is_dir():
                console.print(f"[red]Error: Input path is not a directory: {input_dir}[/red]")
                sys.exit(1)
            
            # Process batch
            process_batch(input_dir, output_dir, args.device, args.confidence, args.action_threshold)
    
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()