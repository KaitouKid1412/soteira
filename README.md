# Real-Time Video Gating System

A production-ready Python 3.10 system for intelligent video frame processing that implements a 3-gate pipeline to filter and send only interesting frames to an LLM (GPT-5).

## Overview

The system processes video input through three consecutive gates:

1. **Motion Gate** (every frame) - Ultra-fast motion detection using MOG2 background subtraction
2. **Scene Change Gate** (on motion + periodic) - HSV histogram and SSIM comparison to detect scene changes  
3. **Object Entry Gate** (stride + triggers) - YOLO object detection with IOU tracking to detect new objects

When significant scene changes or new objects are detected, frames are saved and queued for LLM processing in a background thread.

## Features

- **Real-time processing** optimized for Apple Silicon M2 (16GB RAM)
- **Multi-threaded architecture** with non-blocking LLM processing
- **Adaptive detection cadence** based on motion and scene changes
- **Comprehensive logging** with CSV event logs and frame saving
- **Live preview** with overlays showing gate status and detections
- **Robust error handling** with RTSP reconnection and graceful shutdown

## Installation

### Requirements
- Python 3.10+
- Poetry (for dependency management)
- Apple Silicon Mac (for MPS acceleration) or CPU fallback
- Webcam, video file, or RTSP stream

### Setup with Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone or download the project
cd soteira

# Install dependencies and create virtual environment
poetry install

# Activate the virtual environment
poetry shell

# The system will automatically download YOLOv8n model on first run
```

### Alternative Setup with pip

```bash
# Clone or download the project
cd soteira

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Examples

**Webcam monitoring:**
```bash
# With Poetry
poetry run python main.py --source 0 --show --prompt "Alert if a new object appears or when the scene changes."

# Or if in poetry shell
python main.py --source 0 --show --prompt "Alert if a new object appears or when the scene changes."
```

**Video file processing:**
```bash
poetry run python main.py --source sample.mp4 --save-dir events --det-stride 4 --imgsz 416
```

**RTSP stream monitoring:**
```bash
poetry run python main.py --source "rtsp://user:pass@host:554/stream" --motion-thresh 0.025 --scene-hist 0.32 --scene-ssim 0.64
```

**Dry run (no LLM calls):**
```bash
poetry run python main.py --source 0 --show --dry-run
```

### Command Line Arguments

#### Core Options
- `--source` - Video source: webcam index (0), file path, or RTSP URL
- `--save-dir` - Directory for saving event frames (default: events)
- `--prompt` - Text prompt sent with images to GPT-5
- `--show` - Show live preview with overlays
- `--dry-run` - Log events without calling LLM

#### Motion Gate Parameters
- `--motion-thresh` - Motion threshold as fraction of pixels (default: 0.02)

#### Scene Change Gate Parameters  
- `--scene-hist` - HSV histogram distance threshold (default: 0.30)
- `--scene-ssim` - SSIM similarity threshold (default: 0.65)
- `--scene-safeguard` - Periodic check interval in seconds (default: 2.0)

#### Object Detection Parameters
- `--det-stride` - Run detector every N frames (default: 4)
- `--confirm-hits` - Confirm track after K consecutive hits (default: 2)
- `--max-age` - Track max age in frames (default: 20)
- `--imgsz` - YOLO input size (default: 416)
- `--conf` - Detection confidence threshold (default: 0.4)
- `--iou` - Detection IoU threshold (default: 0.7)
- `--min-area-frac` - Ignore boxes smaller than this fraction (default: 0.01)
- `--classes` - Comma-separated class IDs to keep (e.g., "0,2,5")

## Architecture

### File Structure
```
├── main.py                 # Main entry point and video processing loop
├── gates/
│   ├── motion.py          # Motion detection (MOG2)
│   ├── scene.py           # Scene change detection (HSV + SSIM)
│   └── objects.py         # Object detection and tracking (YOLO + IOU)
├── llm_sink.py            # Background LLM processing thread
├── utils/
│   ├── timing.py          # Performance monitoring utilities
│   └── io.py              # Safe file I/O operations
├── requirements.txt       # Python dependencies
└── events/                # Generated event frames and logs
    ├── events_log.csv     # CSV log of all events
    ├── scene_*.jpg        # Scene change frames
    ├── enter_*.jpg        # Object entry frames
    ├── enter_*_crop.jpg   # Object crops
    └── llm_responses/     # LLM response logs
```

### Processing Pipeline

1. **Frame Capture** - Read from webcam/file/RTSP
2. **Motion Gate** - Every frame, ~5ms at 360p
   - MOG2 background subtraction
   - Morphological cleanup (erode + dilate)
   - Calculate changed pixel ratio
3. **Scene Gate** - On motion + every 2s safeguard
   - HSV histogram comparison (cosine similarity)
   - Grayscale SSIM computation
   - Update reference frame on change
4. **Object Gate** - Every 4th frame + motion spikes + scene changes
   - YOLOv8n detection (~8-15ms on M2/MPS)
   - IOU tracking with confirmation
   - Save frames and crops for new objects
5. **LLM Processing** - Background thread
   - Queue events for GPT-5 processing
   - Non-blocking stub implementation

### Event Types

- **scene_change** - Significant scene change detected
- **object_entered** - New object confirmed in scene

### Logging

**CSV Log Format** (`events/events_log.csv`):
```
ts_iso,event_type,path,track_id_or_blank,d_hist,ssim,changed_ratio
2024-01-15T10:30:45,scene_change,events/scene_1705312245123.jpg,,0.35,0.62,0.025
2024-01-15T10:30:47,object_entered,events/enter_1_1705312247456.jpg,1,,,0.031
```

## Performance Tuning

### For Better Motion Sensitivity
- Decrease `--motion-thresh` (e.g., 0.01 for 1% pixels)
- Increase MOG2 sensitivity in `gates/motion.py`

### For Scene Change Sensitivity  
- Decrease `--scene-hist` (more sensitive to color changes)
- Increase `--scene-ssim` (more sensitive to structural changes)

### For Object Detection Performance
- Increase `--det-stride` to run detection less frequently
- Decrease `--imgsz` for faster inference (e.g., 320)
- Increase `--conf` to reduce false positives

### For Memory Usage
- Implement periodic cleanup in `utils/io.py`
- Reduce `--max-age` for shorter track history
- Compress saved images with lower quality

## Troubleshooting

### RTSP Connection Issues
- Check network connectivity and credentials
- System automatically retries every 2 seconds
- Monitor console for reconnection attempts

### Performance Issues  
- Verify MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Reduce detection frequency with higher `--det-stride`
- Lower resolution input or detection `--imgsz`

### Detection Accuracy
- Adjust `--conf` threshold (lower = more detections)
- Modify `--min-area-frac` to filter small objects
- Use `--classes` to focus on specific object types

## Development

### Extending the System

**Adding New Detection Models:**
- Modify `gates/objects.py` to support additional YOLO variants
- Implement model switching in CLI arguments

**Custom Event Types:**
- Extend event logging in `main.py`
- Add new gate types in the `gates/` package

**LLM Integration:**
- Replace stub in `llm_sink.py` with actual API calls
- Add authentication and error handling

### Poetry Development Workflow

**Managing dependencies:**
```bash
# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree
```

**Virtual environment management:**
```bash
# Create and enter shell
poetry shell

# Run commands without entering shell
poetry run python script.py

# Show environment info
poetry env info

# Deactivate (when in poetry shell)
exit
```

**Development tools:**
```bash
# Code formatting
poetry run black .

# Linting
poetry run flake8 .

# Type checking
poetry run mypy .

# Run tests (when test files are added)
poetry run pytest
```

### Testing

```bash
# Test with webcam
poetry run python main.py --source 0 --show --dry-run

# Test motion sensitivity (wave hand in front of camera)
poetry run python main.py --source 0 --show --motion-thresh 0.01

# Test scene detection (move camera or change lighting)  
poetry run python main.py --source 0 --show --scene-hist 0.2

# Test object detection (walk into frame)
poetry run python main.py --source 0 --show --det-stride 2
```

## License

This project is for defensive security and research purposes only.