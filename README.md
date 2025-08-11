# Video Action Recognition & Tracking Pipeline
## 📌 Overview
This project implements a **real-time video action recognition and tracking system** using **YOLOv8** for object detection, a deep learning–based action recognizer for human activity classification, and a tracking module to maintain identities across frames.
It can process **single videos** or **batch datasets**, making it suitable for surveillance analytics, sports analysis, and human behavior monitoring.
---

## ✨ Features

* 🎯 **YOLOv8-based Object Detection** — Fast and accurate detection of people and objects.
* 🏃 **Action Recognition** — Classifies human activities using Kinetics dataset classes.
* 🔍 **Multi-object Tracking** — Tracks entities across frames for temporal consistency.
* 📂 **Batch Video Processing** — Automates processing of multiple videos.
* 📊 **Progress Logging** — Real-time progress bars and detailed logs.
* ⚙️ **Configurable Settings** — Easily adjust model paths, thresholds, and processing parameters.

---

## 📂 Project Structure

```
├── main.py                     # Entry point for single video processing
├── batch_video_processor.py    # Script for batch processing videos
├── yolov8n.pt                   # Pretrained YOLOv8 weights
├── requirements.txt             # Python dependencies
├── config/
│   ├── kinetics_classes.py      # Action recognition class names
│   ├── settings.py              # Project configuration
├── models/
│   ├── detector.py              # YOLOv8-based object detection
│   ├── action_recognizer.py     # Human activity recognition
│   ├── tracker.py               # Object tracking
├── pipeline/
│   ├── video_processor.py       # Core pipeline for detection + recognition + tracking
├── utils/
│   ├── video_utils.py           # Video reading/writing helpers
│   ├── logger.py                # Logging utilities
│   ├── fast_progress.py         # Progress bar
```

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Single Video Processing

```bash
python main.py --video_path path/to/video.mp4 --output_path results/
```

### Batch Processing

```bash
python batch_video_processor.py --input_dir videos/ --output_dir results/
```

**Command-line arguments** may include:

* `--confidence` — Detection confidence threshold (default: 0.25)
* `--device` — `cpu` or `cuda` for GPU acceleration
* `--classes` — List of class IDs to detect

---

## 📊 Example Output

* Bounding boxes for detected objects
* Action labels displayed over tracked persons
* Saved processed videos in the output directory

---

## 📦 Dependencies

* Python 3.10+
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* tqdm

*(Full list in `requirements.txt`)*

---

## 👤 Author

**Vikram Kumar**
[GitHub Profile](https://github.com/Viki2223) | [LinkedIn](https://www.linkedin.com/in/vikram-kumar-69a4a42a1/)

---

