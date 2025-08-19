"""
Scene change detection gate using HSV histogram and SSIM comparison.
Runs only when motion is detected (threshold-based triggering).
"""

import time
import cv2
import numpy as np
from scipy.signal import convolve2d


class SceneGate:
    def __init__(self, hist_threshold=0.30, ssim_threshold=0.65, min_interval_ms=100):
        """
        Initialize scene change detection gate.
        
        Args:
            hist_threshold: HSV histogram distance threshold for scene change
            ssim_threshold: SSIM threshold for scene change (lower = more change)
            min_interval_ms: Minimum milliseconds between scene checks (throttling)
        """
        self.hist_threshold = hist_threshold
        self.ssim_threshold = ssim_threshold
        self.min_interval_ms = min_interval_ms
        
        # Reference frame storage
        self.s_ref_small = None      # Downscaled reference frame
        self.s_ref_hist = None       # HSV histogram of reference
        self.s_ref_gray = None       # Grayscale reference for SSIM
        
        # Performance optimization state
        self.last_check_time = 0     # Last time scene detection ran
        self.last_result = (False, 0.0, 1.0)  # Cache last result for throttled frames
        self.consecutive_no_change = 0  # Count frames with no change
        
        # SSIM Gaussian kernel (11x11 with sigma=1.5)
        self.ssim_kernel = self._create_gaussian_kernel(11, 1.5)
        
    def _create_gaussian_kernel(self, size, sigma):
        """Create Gaussian kernel for SSIM computation."""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        return kernel / kernel.sum()
    
    def hsv_hist_small(self, img_bgr):
        """
        Compute HSV histogram focusing on H and S channels.
        
        Args:
            img_bgr: BGR image
            
        Returns:
            numpy.ndarray: Normalized histogram (H: 32 bins, S: 32 bins)
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Extract H and S channels (ignore V for lighting invariance)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        
        # Compute 2D histogram: H (0-179) and S (0-255)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        
        # Normalize histogram
        hist_norm = hist / (hist.sum() + 1e-7)
        
        return hist_norm.flatten()
    
    def cosine_similarity(self, a, b):
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a, b: numpy arrays
            
        Returns:
            float: cosine similarity (0-1)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def ssim_gray(self, img1, img2):
        """
        Compute Structural Similarity Index (SSIM) between two grayscale images.
        Inline implementation without external dependencies.
        
        Args:
            img1, img2: numpy arrays (grayscale images, same size)
            
        Returns:
            float: SSIM value (0-1, higher = more similar)
        """
        if img1.shape != img2.shape:
            return 0.0
        
        # Convert to float
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # SSIM constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Gaussian filtering for local means
        mu1 = convolve2d(img1, self.ssim_kernel, mode='valid', boundary='symm')
        mu2 = convolve2d(img2, self.ssim_kernel, mode='valid', boundary='symm')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Local variances and covariance
        sigma1_sq = convolve2d(img1 ** 2, self.ssim_kernel, mode='valid', boundary='symm') - mu1_sq
        sigma2_sq = convolve2d(img2 ** 2, self.ssim_kernel, mode='valid', boundary='symm') - mu2_sq
        sigma12 = convolve2d(img1 * img2, self.ssim_kernel, mode='valid', boundary='symm') - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-7)
        
        # Return mean SSIM
        return np.mean(ssim_map)
    
    
    def check_scene_change(self, curr_small):
        """
        Check if current frame represents a scene change.
        
        Args:
            curr_small: Current frame (downscaled, BGR)
            
        Returns:
            tuple: (is_change, d_hist, ssim)
                - is_change: bool indicating scene change
                - d_hist: histogram distance (0-1)
                - ssim: SSIM similarity (0-1)
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Initialize reference if not set
        if self.s_ref_small is None:
            self._update_reference(curr_small)
            self.last_check_time = current_time
            return False, 0.0, 1.0
        
        # Performance optimization: throttle scene detection
        time_since_last = current_time - self.last_check_time
        
        # Skip expensive computation if called too frequently
        if time_since_last < self.min_interval_ms:
            # Return cached result but don't trigger scene change
            is_change, d_hist, ssim = self.last_result
            return False, d_hist, ssim  # Force no change to avoid excessive triggers
        
        # Update timing
        self.last_check_time = current_time
        
        # Compute current frame features
        curr_hist = self.hsv_hist_small(curr_small)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        
        # HSV histogram comparison
        cosine_sim = self.cosine_similarity(curr_hist, self.s_ref_hist)
        d_hist = 1.0 - cosine_sim  # Distance: 0=identical, 1=completely different
        
        # SSIM comparison
        ssim_val = self.ssim_gray(curr_gray, self.s_ref_gray)
        
        # Scene change detection
        hist_change = d_hist > self.hist_threshold
        ssim_change = ssim_val < self.ssim_threshold
        
        is_change = hist_change or ssim_change
        
        # Update reference and tracking
        if is_change:
            self._update_reference(curr_small)
            self.consecutive_no_change = 0
        else:
            self.consecutive_no_change += 1
            
            # Adaptive throttling: increase interval if no changes detected
            if self.consecutive_no_change > 10:
                self.min_interval_ms = min(self.min_interval_ms * 1.2, 500)  # Max 500ms
        
        # Cache result
        self.last_result = (is_change, d_hist, ssim_val)
        
        return is_change, d_hist, ssim_val
    
    def _update_reference(self, frame_small):
        """
        Update reference frame and its features.
        
        Args:
            frame_small: New reference frame (BGR)
        """
        self.s_ref_small = frame_small.copy()
        self.s_ref_hist = self.hsv_hist_small(frame_small)
        self.s_ref_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    
    def reset_reference(self, frame_small=None):
        """
        Reset reference frame (useful for manual reinitialization).
        
        Args:
            frame_small: Optional new reference frame
        """
        if frame_small is not None:
            self._update_reference(frame_small)
        else:
            self.s_ref_small = None
            self.s_ref_hist = None
            self.s_ref_gray = None
        
        # Reset timing and throttling state
        self.last_check_time = time.time() * 1000
        self.last_result = (False, 0.0, 1.0)
        self.consecutive_no_change = 0