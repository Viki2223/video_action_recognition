import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for d in ["Datasets/InputVideos", "Datasets/output_videos",
          "Datasets/cropped_persons", "Datasets/extracted_frames",
          "Datasets/logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)

CONFIG = {
    "yolo_model": "yolov8n.pt",
    "action_model": "r3d_18.pt",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.5,
    "person_class_id": 0,
    "max_age": 20,
    "min_hits": 2,
    "iou_threshold_track": 0.3,
    "sequence_length": 8,
    "action_confidence_threshold": 0.3,
    "target_fps": 30,
    "resize_height": 384,
    "resize_width": 640,
    "input_dir": str(BASE_DIR / "Datasets/InputVideos"),
    "output_dir": str(BASE_DIR / "Datasets/output_videos"),
    "crops_dir": str(BASE_DIR / "Datasets/cropped_persons"),
    "frames_dir": str(BASE_DIR / "Datasets/extracted_frames"),
    "logs_dir": str(BASE_DIR / "Datasets/logs"),
    "min_box_area": 200,
    "max_box_area": 300000,
    "aspect_ratio_threshold": 0.2,
    "process_every_nth_frame": 1,
    "resize_factor": 1.0,
    "action_skip_frames": 1,
    "action_classes": [
        "walking", "running", "standing", "sitting", "jumping",
        "waving", "clapping", "dancing", "climbing", "exercising",
        "talking", "eating", "drinking", "reading", "working",
        "playing", "stretching", "unknown",
    ],
}

ACTION_COLORS = {a: (255, 255, 255) for a in CONFIG["action_classes"]}
ACTION_COLORS.update({
    "walking": (0, 255, 0),
    "running": (0, 0, 255),
    "standing": (255, 0, 0),
    "sitting": (255, 255, 0),
    "jumping": (255, 0, 255),
    "waving": (0, 255, 255),
    "clapping": (128, 255, 0),
    "dancing": (255, 128, 0),
    "unknown": (128, 128, 128),
})