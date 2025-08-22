#!/usr/bin/env python3
"""
Video file streamer that simulates real-time camera input.
Reads a video file and serves frames at the original frame rate.
"""

import cv2
import time
import threading
import queue
from pathlib import Path
from typing import Optional


class VideoStreamer:
    """Stream video file as real-time camera feed."""
    
    def __init__(self, video_path: str, loop: bool = True, speed_multiplier: float = 1.0):
        """
        Initialize video streamer.
        
        Args:
            video_path: Path to video file
            loop: Whether to loop the video when it ends
            speed_multiplier: Speed control (1.0 = normal, 2.0 = 2x speed, 0.5 = half speed)
        """
        self.video_path = Path(video_path)
        self.loop = loop
        self.speed_multiplier = speed_multiplier
        
        # Video properties
        self.cap = None
        self.fps = 30.0  # Default FPS, will be updated from video
        self.frame_interval = 1.0 / 30.0  # Time between frames
        
        # Streaming state
        self.frame_queue = queue.Queue(maxsize=100)  # Buffer for smooth playback
        self.streaming = False
        self.streamer_thread = None
        
        # Statistics
        self.frames_streamed = 0
        self.start_time = None
        
        # Initialize video capture
        self._initialize_video()
    
    def _initialize_video(self):
        """Initialize video capture and get properties."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / self.fps
        
        # Calculate frame interval with speed multiplier
        self.frame_interval = (1.0 / self.fps) / self.speed_multiplier
        
        print(f"[STREAMER] Video: {self.video_path.name}")
        print(f"[STREAMER] Resolution: {width}x{height}")
        print(f"[STREAMER] FPS: {self.fps:.1f}")
        print(f"[STREAMER] Frames: {frame_count}")
        print(f"[STREAMER] Duration: {duration:.1f}s")
        print(f"[STREAMER] Speed: {self.speed_multiplier}x")
        print(f"[STREAMER] Frame interval: {self.frame_interval*1000:.1f}ms")
    
    def start_streaming(self):
        """Start streaming video frames."""
        if self.streaming:
            return
        
        self.streaming = True
        self.start_time = time.time()
        self.frames_streamed = 0
        
        # Start background streaming thread
        self.streamer_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.streamer_thread.start()
        
        print(f"[STREAMER] Started streaming at {self.fps*self.speed_multiplier:.1f} FPS")
    
    def stop_streaming(self):
        """Stop streaming video frames."""
        self.streaming = False
        
        if self.streamer_thread and self.streamer_thread.is_alive():
            self.streamer_thread.join(timeout=2)
        
        # Clear remaining frames from queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frames_streamed / elapsed if elapsed > 0 else 0
        
        print(f"[STREAMER] Stopped streaming")
        print(f"[STREAMER] Streamed {self.frames_streamed} frames in {elapsed:.1f}s")
        print(f"[STREAMER] Average FPS: {avg_fps:.1f}")
    
    def _stream_loop(self):
        """Main streaming loop running in background thread."""
        last_frame_time = time.time()
        
        while self.streaming:
            # Read next frame
            ret, frame = self.cap.read()
            
            if not ret:
                if self.loop:
                    # Restart video from beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print(f"[STREAMER] Looping video (streamed {self.frames_streamed} frames)")
                    continue
                else:
                    # End of video, stop streaming
                    print(f"[STREAMER] End of video reached")
                    self.streaming = False
                    break
            
            # Wait for correct frame timing
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            
            if time_since_last < self.frame_interval:
                sleep_time = self.frame_interval - time_since_last
                time.sleep(sleep_time)
            
            # Add frame to queue (non-blocking, drop if queue is full)
            try:
                self.frame_queue.put(frame, block=False)
                self.frames_streamed += 1
                last_frame_time = time.time()
                
            except queue.Full:
                # Drop frame if queue is full (consumer too slow)
                queue_size = self.frame_queue.qsize()
                elapsed = time.time() - self.start_time if self.start_time else 0
                actual_fps = self.frames_streamed / elapsed if elapsed > 0 else 0
                
                # Only print drop message every 50 frames to reduce spam
                if self.frames_streamed % 50 == 0:
                    print(f"[STREAMER DROP] Frame {self.frames_streamed} dropped - Queue full ({queue_size}/100), actual FPS: {actual_fps:.1f}, target: {self.fps:.1f}")
                
                # Skip multiple frames when queue is consistently full
                if queue_size > 80:  # Queue is 80% full
                    # Skip ahead to reduce backlog
                    for _ in range(min(5, queue_size - 70)):
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
    
    def read(self):
        """
        Read next frame (mimics cv2.VideoCapture.read()).
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if not self.streaming:
            return False, None
        
        try:
            # Get frame from queue with timeout
            frame = self.frame_queue.get(timeout=0.1)
            return True, frame
        except queue.Empty:
            # No frame available
            return False, None
    
    def isOpened(self):
        """Check if streamer is active (mimics cv2.VideoCapture.isOpened())."""
        return self.streaming and (self.cap is not None and self.cap.isOpened())
    
    def release(self):
        """Release resources."""
        self.stop_streaming()
        if self.cap:
            self.cap.release()
    
    def get_stats(self):
        """Get streaming statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "frames_streamed": self.frames_streamed,
            "elapsed_seconds": elapsed,
            "target_fps": self.fps * self.speed_multiplier,
            "actual_fps": self.frames_streamed / elapsed if elapsed > 0 else 0,
            "queue_size": self.frame_queue.qsize(),
            "streaming": self.streaming
        }


def create_test_stream(video_path: str, output_device: int = 1, speed: float = 1.0):
    """
    Create a virtual camera device from video file (requires additional setup).
    This is a placeholder for more advanced virtual camera creation.
    
    Args:
        video_path: Path to video file
        output_device: Virtual camera device number
        speed: Playback speed multiplier
    """
    print(f"[STREAMER] To create virtual camera device:")
    print(f"1. Install OBS Studio with Virtual Camera plugin")
    print(f"2. Add Video Source: {video_path}")
    print(f"3. Start Virtual Camera on device {output_device}")
    print(f"4. Use device {output_device} as source in your application")


def main():
    """Test the video streamer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream video file as real-time camera feed")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true", help="Don't loop video")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Create streamer
    streamer = VideoStreamer(
        video_path=args.video_path,
        loop=not args.no_loop,
        speed_multiplier=args.speed
    )
    
    try:
        # Start streaming
        streamer.start_streaming()
        
        # Read frames for test duration
        start_time = time.time()
        frame_count = 0
        
        print(f"\\n[TEST] Reading frames for {args.duration}s...")
        
        while time.time() - start_time < args.duration:
            ret, frame = streamer.read()
            if ret:
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    stats = streamer.get_stats()
                    print(f"[TEST] Frame {frame_count}, "
                          f"Queue: {stats['queue_size']}, "
                          f"FPS: {stats['actual_fps']:.1f}")
            else:
                time.sleep(0.01)  # Small delay if no frame available
        
        # Show final stats
        stats = streamer.get_stats()
        print(f"\\n[TEST] Final stats:")
        print(f"  Frames read: {frame_count}")
        print(f"  Frames streamed: {stats['frames_streamed']}")
        print(f"  Target FPS: {stats['target_fps']:.1f}")
        print(f"  Actual FPS: {stats['actual_fps']:.1f}")
        
    finally:
        streamer.release()


if __name__ == "__main__":
    main()