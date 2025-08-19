# Video Streaming Guide

This guide shows you how to stream a video file as if it's a real-time camera feed, simulating real-time processing for testing and development.

## Usage

### Basic Video Streaming

Stream a video file at its original frame rate:

```bash
poetry run python main.py --source my_video.mp4 --stream-video --dry-run
```

### Speed Control

Control playback speed to simulate different scenarios:

```bash
# 2x speed (faster than real-time)
poetry run python main.py --source my_video.mp4 --stream-video --stream-speed 2.0 --dry-run

# Half speed (slower than real-time) 
poetry run python main.py --source my_video.mp4 --stream-video --stream-speed 0.5 --dry-run

# 10x speed (very fast testing)
poetry run python main.py --source my_video.mp4 --stream-video --stream-speed 10.0 --dry-run
```

### Loop Control

Control whether video loops when it reaches the end:

```bash
# Loop video (default behavior)
poetry run python main.py --source my_video.mp4 --stream-video --stream-loop --dry-run

# Don't loop - stop when video ends
poetry run python main.py --source my_video.mp4 --stream-video --no-stream-loop --dry-run
```

### Real Processing Examples

Stream video with actual LLM processing:

```bash
# Stream with OpenAI processing
poetry run python main.py --source my_video.mp4 --stream-video --openai-api-key YOUR_KEY --prompt "find humans in the video"

# Stream with parallel processing
poetry run python main.py --source my_video.mp4 --stream-video --llm-workers 8 --openai-api-key YOUR_KEY

# Stream with sensitive motion detection
poetry run python main.py --source my_video.mp4 --stream-video --motion-thresh 0.001 --show
```

## How It Works

### Video Streamer

The `VideoStreamer` class:
- Reads video file and extracts original frame rate
- Streams frames at controlled timing (respects original FPS)
- Buffers frames in a queue for smooth playback
- Handles looping and speed control
- Provides statistics on streaming performance

### Integration

When `--stream-video` is used:
1. System detects it's a video file (not webcam/RTSP)
2. Creates `VideoStreamer` instead of `cv2.VideoCapture`
3. Streams frames at real-time rate with specified speed multiplier
4. Processing pipeline works exactly the same as with live camera

### Performance Considerations

**Frame Dropping**: If processing is slower than stream rate, frames will be dropped:
```
[STREAMER] Dropped frame 15 (queue full)
```

**Solutions**:
- Reduce stream speed: `--stream-speed 0.5`
- Increase processing efficiency
- Use faster hardware
- Reduce detection frequency with higher thresholds

### Timing Accuracy

The streamer maintains accurate timing:
- Original video: 30 FPS = 33.3ms between frames
- 2x speed: 60 FPS = 16.7ms between frames  
- 0.5x speed: 15 FPS = 66.7ms between frames

## Use Cases

### Development & Testing
```bash
# Test with known video content
poetry run python main.py --source test_video.mp4 --stream-video --dry-run --show

# Rapid testing at high speed
poetry run python main.py --source long_video.mp4 --stream-video --stream-speed 10.0 --dry-run
```

### Demo & Presentation
```bash
# Live demo with pre-recorded content
poetry run python main.py --source demo_video.mp4 --stream-video --show --openai-api-key KEY
```

### Performance Analysis
```bash
# Test system performance under load
poetry run python main.py --source stress_test.mp4 --stream-video --stream-speed 5.0 --llm-workers 16
```

### Algorithm Tuning
```bash
# Fine-tune thresholds with consistent input
poetry run python main.py --source reference_video.mp4 --stream-video --motion-thresh 0.01 --scene-hist 0.3
```

## Output

The system provides streaming statistics:

```
[STREAMER] Video: my_video.mp4
[STREAMER] Resolution: 1920x1080  
[STREAMER] FPS: 30.0
[STREAMER] Duration: 120.5s
[STREAMER] Speed: 2.0x
[STREAMER] Frame interval: 16.7ms
[STREAMER] Started streaming at 60.0 FPS

# At shutdown:
[STREAMER] Final stats: 3600 frames, 58.2 FPS avg
```

This allows you to test your video processing pipeline with any video file as if it were a live camera feed!