"""
Smart frame sampling for high FPS video processing.
"""

import time
from typing import Optional, Tuple
import numpy as np


class AdaptiveFrameSampler:
    """Adaptive frame sampler that adjusts processing rate based on performance."""
    
    def __init__(self, target_fps: float = 30.0, min_fps: float = 10.0, max_fps: float = 60.0):
        """
        Initialize adaptive frame sampler.
        
        Args:
            target_fps: Target processing FPS to maintain
            min_fps: Minimum processing FPS (always process at least this rate)
            max_fps: Maximum processing FPS (cap processing rate)
        """
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        
        # Timing tracking
        self.last_process_time = 0
        self.processing_times = []
        self.frame_interval = 1.0 / target_fps
        self.min_interval = 1.0 / max_fps
        self.max_interval = 1.0 / min_fps
        
        # Adaptive state
        self.current_skip_rate = 1  # Process every N frames
        self.frame_counter = 0
        self.performance_window = 10  # Frames to average for performance calculation
        
        # Drop reason tracking
        self.last_drop_reason = "No drops yet"
        
    def should_process_frame(self) -> bool:
        """
        Determine if current frame should be processed based on timing and performance.
        
        Returns:
            bool: True if frame should be processed
        """
        self.frame_counter += 1
        current_time = time.time()
        
        # Always process first frame
        if self.frame_counter == 1:
            self.last_process_time = current_time
            return True
        
        # Check if enough time has passed since last processing
        time_since_last = current_time - self.last_process_time
        
        # Adaptive skipping based on recent performance
        if len(self.processing_times) >= 3:
            avg_processing_time = np.mean(self.processing_times[-5:])
            
            # If processing is taking too long, increase skip rate
            if avg_processing_time > self.frame_interval * 1.5:
                self.current_skip_rate = min(self.current_skip_rate + 1, 6)
            # If processing is fast, decrease skip rate
            elif avg_processing_time < self.frame_interval * 0.5:
                self.current_skip_rate = max(self.current_skip_rate - 1, 1)
        
        # Process frame if skip interval is met and minimum time has passed
        if (self.frame_counter % self.current_skip_rate == 0 and 
            time_since_last >= self.min_interval):
            self.last_process_time = current_time
            return True
        
        # Always process if too much time has passed (ensure minimum FPS)
        if time_since_last >= self.max_interval:
            self.last_process_time = current_time
            return True
        
        # Record why frame was dropped
        if self.frame_counter % self.current_skip_rate != 0:
            avg_proc_ms = f"{np.mean(self.processing_times[-3:]):.1f}" if len(self.processing_times) >= 3 else "N/A"
            self.last_drop_reason = f"Adaptive skip (rate={self.current_skip_rate}, avg_proc_time={avg_proc_ms}ms)"
        elif time_since_last < self.min_interval:
            self.last_drop_reason = f"Rate limiting (time_since_last={time_since_last*1000:.1f}ms < min_interval={self.min_interval*1000:.1f}ms)"
        else:
            self.last_drop_reason = "Unknown reason"
        
        return False
    
    def record_processing_time(self, processing_time: float):
        """Record time taken to process a frame."""
        self.processing_times.append(processing_time * 1000)  # Convert to ms
        
        # Keep only recent times
        if len(self.processing_times) > self.performance_window:
            self.processing_times = self.processing_times[-self.performance_window:]
    
    def get_last_drop_reason(self) -> str:
        """Get the reason why the last frame was dropped."""
        return self.last_drop_reason
    
    def get_stats(self) -> dict:
        """Get current sampling statistics."""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        effective_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "current_skip_rate": self.current_skip_rate,
            "frames_seen": self.frame_counter,
            "frames_processed": len(self.processing_times),
            "avg_processing_time_ms": avg_time * 1000,
            "effective_fps": effective_fps,
            "target_fps": self.target_fps
        }


class MotionBasedSampler:
    """Frame sampler that processes more frames when motion is detected."""
    
    def __init__(self, base_fps: float = 15.0, motion_fps: float = 30.0):
        """
        Initialize motion-based sampler.
        
        Args:
            base_fps: FPS when no motion detected
            motion_fps: FPS when motion detected
        """
        self.base_fps = base_fps
        self.motion_fps = motion_fps
        self.base_interval = 1.0 / base_fps
        self.motion_interval = 1.0 / motion_fps
        
        self.last_process_time = 0
        self.motion_detected = False
        self.motion_decay_time = 2.0  # Seconds to maintain high FPS after motion
        self.last_motion_time = 0
        
    def should_process_frame(self, motion_detected: bool = False) -> bool:
        """
        Determine if frame should be processed based on motion state.
        
        Args:
            motion_detected: Whether motion was detected in current or recent frames
            
        Returns:
            bool: True if frame should be processed
        """
        current_time = time.time()
        
        # Update motion state
        if motion_detected:
            self.motion_detected = True
            self.last_motion_time = current_time
        else:
            # Check if motion period has expired
            if current_time - self.last_motion_time > self.motion_decay_time:
                self.motion_detected = False
        
        # Determine processing interval based on motion state
        required_interval = self.motion_interval if self.motion_detected else self.base_interval
        
        # Check if enough time has passed
        time_since_last = current_time - self.last_process_time
        
        if time_since_last >= required_interval:
            self.last_process_time = current_time
            return True
        
        return False
    
    def get_current_mode(self) -> str:
        """Get current sampling mode."""
        return "motion" if self.motion_detected else "base"