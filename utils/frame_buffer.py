"""
Frame buffer system for capturing high-quality frames after event detection.
"""

import time
from collections import deque
import numpy as np
from .image_quality import is_image_quality_good, calculate_blur_score


class PendingCapture:
    """Represents a pending frame capture operation."""
    
    def __init__(self, event_type, track_id=None, frames_to_capture=10, 
                 blur_threshold=25.0, min_brightness=20.0, max_brightness=240.0, min_contrast=10.0, 
                 disable_quality_filter=False, stop_on_good_frame=True, start_frame_number=0):
        self.event_type = event_type
        self.track_id = track_id
        self.frames_to_capture = frames_to_capture
        self.frames_captured = 0
        self.start_time = time.time()
        self.start_frame_number = start_frame_number
        
        # Quality thresholds
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.disable_quality_filter = disable_quality_filter
        self.stop_on_good_frame = stop_on_good_frame
        
        # Store captured frames with quality scores
        self.candidate_frames = []  # List of (frame, quality_score, timestamp)
        
        # Track which specific frame numbers this capture will process
        self.target_frame_numbers = set(range(start_frame_number + 1, start_frame_number + 1 + frames_to_capture))
        
    def add_frame(self, frame, frame_number):
        """Add a frame candidate and evaluate its quality."""
        if self.frames_captured >= self.frames_to_capture:
            return False  # Done capturing
        
        # Check if this frame number is in our target range
        if frame_number not in self.target_frame_numbers:
            return True  # Continue waiting for our target frames
        
        # Calculate comprehensive quality score
        is_good_quality, quality_metrics = is_image_quality_good(
            frame, self.blur_threshold, self.min_brightness, self.max_brightness, self.min_contrast
        )
        
        # Combined quality score (higher is better)
        quality_score = (
            quality_metrics['blur_score'] * 1.0 +      # Blur is most important
            quality_metrics['contrast'] * 0.5 +        # Contrast second
            (100 - abs(quality_metrics['brightness'] - 127.5)) * 0.3  # Brightness around middle
        )
        
        timestamp = time.time()
        self.candidate_frames.append((frame.copy(), quality_score, timestamp, quality_metrics))
        self.frames_captured += 1
        
        # Debug: Print quality metrics
        print(f"[BUFFER] Frame {self.frames_captured}/{self.frames_to_capture} for {self.event_type}: "
              f"blur={quality_metrics['blur_score']:.1f}, "
              f"brightness={quality_metrics['brightness']:.1f}, "
              f"contrast={quality_metrics['contrast']:.1f}, "
              f"quality_score={quality_score:.1f}")
        
        # Early exit: If we found a good quality frame and stop_on_good_frame is enabled
        if is_good_quality and self.stop_on_good_frame:
            print(f"[BUFFER] ✓ Found good quality frame for {self.event_type}, stopping capture early")
            return False  # Stop capturing
        
        return self.frames_captured < self.frames_to_capture
    
    def get_best_frame(self):
        """Get the highest quality frame from candidates."""
        if not self.candidate_frames:
            return None, None
        
        # Sort by quality score (highest first)
        self.candidate_frames.sort(key=lambda x: x[1], reverse=True)
        
        # Return best frame and its metrics
        best_frame, best_score, timestamp, metrics = self.candidate_frames[0]
        
        # Check if best frame meets minimum quality standards
        is_good_quality, quality_check = is_image_quality_good(
            best_frame, self.blur_threshold, self.min_brightness, 
            self.max_brightness, self.min_contrast
        )
        
        print(f"[BUFFER] Best frame for {self.event_type}: "
              f"blur={metrics['blur_score']:.1f} (need {self.blur_threshold}), "
              f"brightness={metrics['brightness']:.1f} (need {self.min_brightness}-{self.max_brightness}), "
              f"contrast={metrics['contrast']:.1f} (need {self.min_contrast})")
        
        if is_good_quality or self.disable_quality_filter:
            status = "✓ Frame quality acceptable" if is_good_quality else "⚠ Quality filter disabled, using best available"
            print(f"[BUFFER] {status}")
            return best_frame, metrics
        else:
            print(f"[BUFFER] ✗ Frame quality too low - "
                  f"sharp={quality_check['is_sharp']}, "
                  f"lit={quality_check['is_well_lit']}, "
                  f"contrast_ok={quality_check['has_contrast']}")
            return None, None
    
    def is_complete(self):
        """Check if capture is complete."""
        return self.frames_captured >= self.frames_to_capture
    
    def get_age(self):
        """Get age of this capture in seconds."""
        return time.time() - self.start_time


class FrameBuffer:
    """Buffer system to capture high-quality frames after events."""
    
    def __init__(self, max_pending=10, capture_timeout=5.0):
        self.max_pending = max_pending
        self.capture_timeout = capture_timeout
        self.pending_captures = []
        
        # Frame deduplication: track which frames are already being captured
        self.frame_capture_schedule = set()  # Set of frame numbers being captured
        self.current_frame_number = 0
    
    def request_capture(self, event_type, track_id=None, frames_to_capture=10,
                       blur_threshold=25.0, min_brightness=20.0, max_brightness=240.0, min_contrast=10.0,
                       disable_quality_filter=False, stop_on_good_frame=True):
        """Request a new frame capture operation with overlap prevention."""
        
        # Prevent too many pending captures
        if len(self.pending_captures) >= self.max_pending:
            print(f"[BUFFER] Too many pending captures, skipping {event_type}")
            return False
        
        # Calculate proposed frame range
        start_frame = self.current_frame_number
        proposed_frames = set(range(start_frame + 1, start_frame + 1 + frames_to_capture))
        
        # Check for overlap with existing captures
        overlap = proposed_frames.intersection(self.frame_capture_schedule)
        if overlap:
            # Adjust the capture to avoid overlap
            available_frames = proposed_frames - self.frame_capture_schedule
            if len(available_frames) < max(1, frames_to_capture // 3):  # Need at least 1/3 of requested frames
                print(f"[BUFFER] Skipping {event_type} - too much overlap with existing captures")
                return False
            
            # Adjust frames_to_capture to available frames
            frames_to_capture = min(frames_to_capture, len(available_frames))
            print(f"[BUFFER] Adjusted capture for {event_type} to avoid overlap: {frames_to_capture} frames")
        
        # Create new pending capture
        capture = PendingCapture(
            event_type, track_id, frames_to_capture,
            blur_threshold, min_brightness, max_brightness, min_contrast,
            disable_quality_filter, stop_on_good_frame, start_frame
        )
        
        # Add frame numbers to schedule
        self.frame_capture_schedule.update(capture.target_frame_numbers)
        
        self.pending_captures.append(capture)
        print(f"[BUFFER] Started capture for {event_type} (track_id={track_id}), frames {min(capture.target_frame_numbers)}-{max(capture.target_frame_numbers)}")
        return True
    
    def process_frame(self, frame):
        """Process current frame for all pending captures. Frame can be None during shutdown."""
        if frame is not None:
            self.current_frame_number += 1
        completed_captures = []
        
        # Update all pending captures
        for capture in self.pending_captures[:]:  # Copy list to allow modification
            
            # Check for timeout
            if capture.get_age() > self.capture_timeout:
                print(f"[BUFFER] Capture {capture.event_type} timed out after {capture.get_age():.1f}s")
                # Remove frame numbers from schedule
                self.frame_capture_schedule -= capture.target_frame_numbers
                completed_captures.append(capture)
                self.pending_captures.remove(capture)
                continue
            
            # Add frame to capture (skip if frame is None during shutdown)
            if frame is not None:
                still_capturing = capture.add_frame(frame, self.current_frame_number)
            else:
                # During shutdown, just check if capture is complete based on timeout
                still_capturing = False
            
            # If capture is complete, move to completed
            if not still_capturing:
                # Remove frame numbers from schedule
                self.frame_capture_schedule -= capture.target_frame_numbers
                completed_captures.append(capture)
                self.pending_captures.remove(capture)
        
        return completed_captures
    
    def get_pending_count(self):
        """Get number of pending captures."""
        return len(self.pending_captures)
    
    def get_pending_info(self):
        """Get info about pending captures for debugging."""
        info = []
        for capture in self.pending_captures:
            info.append({
                'event_type': capture.event_type,
                'track_id': capture.track_id,
                'frames_captured': capture.frames_captured,
                'frames_to_capture': capture.frames_to_capture,
                'age': capture.get_age()
            })
        return info
    
    def get_schedule_info(self):
        """Get information about the frame capture schedule."""
        if not self.frame_capture_schedule:
            return "No frames scheduled"
        
        min_frame = min(self.frame_capture_schedule)
        max_frame = max(self.frame_capture_schedule)
        return f"Scheduled frames: {min_frame}-{max_frame} ({len(self.frame_capture_schedule)} total)"
    
    def clear_old_captures(self):
        """Clear captures that have timed out."""
        current_time = time.time()
        old_captures = [
            capture for capture in self.pending_captures 
            if capture.get_age() >= self.capture_timeout
        ]
        
        # Remove frame numbers from schedule for old captures
        for capture in old_captures:
            self.frame_capture_schedule -= capture.target_frame_numbers
        
        self.pending_captures = [
            capture for capture in self.pending_captures 
            if capture.get_age() < self.capture_timeout
        ]
        
        # Also clean up very old frame numbers from schedule (older than 100 frames)
        old_frame_threshold = self.current_frame_number - 100
        self.frame_capture_schedule = {
            frame_num for frame_num in self.frame_capture_schedule 
            if frame_num > old_frame_threshold
        }