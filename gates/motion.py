"""
Motion detection gate using MOG2 background subtraction.
Runs on every frame to detect motion spikes.
"""

import cv2
import numpy as np


class MotionGate:
    def __init__(self, threshold=0.02):
        """
        Initialize motion detection gate.
        
        Args:
            threshold: Motion threshold (fraction of pixels that must change)
        """
        self.threshold = threshold
        
        # MOG2 background subtractor
        # history=300: number of frames for background model
        # varThreshold=16: threshold on squared Mahalanobis distance
        # detectShadows=False: disable shadow detection for speed
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=16,
            detectShadows=False
        )
        
        # Morphological kernels for cleanup
        # Erode to remove noise, then dilate to fill gaps
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        self.frame_count = 0
    
    def apply_motion(self, frame_bgr):
        """
        Apply motion detection to a frame.
        
        Args:
            frame_bgr: Input frame in BGR format (should be downscaled ~640x360)
            
        Returns:
            tuple: (changed_ratio, motion_mask)
                - changed_ratio: fraction of pixels with motion (0.0-1.0)
                - motion_mask: binary mask of motion areas
        """
        self.frame_count += 1
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame_bgr)
        
        # Morphological cleanup to reduce noise
        # First erode to remove small noise
        cleaned = cv2.erode(fg_mask, self.erode_kernel, iterations=1)
        # Then dilate to connect nearby regions
        cleaned = cv2.dilate(cleaned, self.dilate_kernel, iterations=1)
        
        # Calculate motion ratio
        total_pixels = cleaned.shape[0] * cleaned.shape[1]
        foreground_pixels = cv2.countNonZero(cleaned)
        changed_ratio = foreground_pixels / total_pixels
        
        return changed_ratio, cleaned
    
    def is_motion_spike(self, changed_ratio):
        """
        Check if the current frame has a motion spike.
        
        Args:
            changed_ratio: fraction of pixels with motion
            
        Returns:
            bool: True if motion exceeds threshold
        """
        return changed_ratio > self.threshold
    
    def reset_background(self):
        """Reset the background model (useful for scene changes)."""
        # Create new background subtractor to reset the model
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=16,
            detectShadows=False
        )


class SimpleMotionGate:
    """
    Alternative implementation using simple frame differencing.
    Faster but less robust than MOG2.
    """
    
    def __init__(self, threshold=0.02):
        self.threshold = threshold
        self.prev_frame = None
        
        # Morphological kernels
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    def apply_motion(self, frame_bgr):
        """
        Apply simple frame differencing for motion detection.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            tuple: (changed_ratio, motion_mask)
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0, np.zeros_like(gray)
        
        # Frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        
        # Threshold to binary
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        cleaned = cv2.erode(thresh, self.erode_kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, self.dilate_kernel, iterations=1)
        
        # Calculate motion ratio
        total_pixels = cleaned.shape[0] * cleaned.shape[1]
        foreground_pixels = cv2.countNonZero(cleaned)
        changed_ratio = foreground_pixels / total_pixels
        
        # Update previous frame
        self.prev_frame = gray
        
        return changed_ratio, cleaned
    
    def is_motion_spike(self, changed_ratio):
        return changed_ratio > self.threshold