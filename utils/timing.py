"""
Timing utilities for performance monitoring.
"""

import time
from collections import deque


class FPSCounter:
    """Real-time FPS counter with smoothing."""
    
    def __init__(self, window_size=30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        
    def tick(self):
        """Record a frame tick."""
        current_time = time.time()
        
        if len(self.frame_times) == 0:
            # First frame
            self.frame_times.append(current_time)
        else:
            # Record time difference
            self.frame_times.append(current_time)
        
        self.last_time = current_time
    
    def get_fps(self):
        """
        Get current FPS.
        
        Returns:
            float: Frames per second
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate FPS from time differences
        total_time = self.frame_times[-1] - self.frame_times[0]
        num_intervals = len(self.frame_times) - 1
        
        if total_time <= 0 or num_intervals <= 0:
            return 0.0
        
        return num_intervals / total_time
    
    def reset(self):
        """Reset the FPS counter."""
        self.frame_times.clear()
        self.last_time = time.time()


class PerformanceTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name="operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        print(f"[TIMER] {self.name}: {elapsed*1000:.2f}ms")
    
    def elapsed_ms(self):
        """Get elapsed time in milliseconds."""
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class RateLimit:
    """Rate limiter for operations."""
    
    def __init__(self, max_calls_per_second):
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_second: Maximum allowed calls per second
        """
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def can_proceed(self):
        """
        Check if operation can proceed without waiting.
        
        Returns:
            bool: True if operation can proceed immediately
        """
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        return time_since_last >= self.min_interval


class MovingAverage:
    """Moving average calculator."""
    
    def __init__(self, window_size=10):
        """
        Initialize moving average.
        
        Args:
            window_size: Number of values to average over
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value):
        """Add a new value and return current average."""
        self.values.append(value)
        return self.get_average()
    
    def get_average(self):
        """Get current average."""
        if len(self.values) == 0:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the moving average."""
        self.values.clear()


def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = (end_time - start_time) * 1000
        print(f"[TIMER] {func.__name__}: {elapsed:.2f}ms")
        return result
    return wrapper


def get_timestamp():
    """Get current timestamp as string."""
    return time.strftime("%Y%m%d_%H%M%S")


def get_iso_timestamp():
    """Get current timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class FrequencyCounter:
    """Count occurrences over time windows."""
    
    def __init__(self, window_seconds=60):
        """
        Initialize frequency counter.
        
        Args:
            window_seconds: Time window for counting
        """
        self.window_seconds = window_seconds
        self.timestamps = deque()
    
    def add_occurrence(self, timestamp=None):
        """Add an occurrence at the given timestamp."""
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        self._cleanup_old()
    
    def get_count(self):
        """Get count of occurrences in the current window."""
        self._cleanup_old()
        return len(self.timestamps)
    
    def get_rate_per_minute(self):
        """Get rate per minute in the current window."""
        count = self.get_count()
        if self.window_seconds == 0:
            return 0.0
        return count * (60.0 / self.window_seconds)
    
    def _cleanup_old(self):
        """Remove timestamps outside the window."""
        cutoff_time = time.time() - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
    
    def reset(self):
        """Reset the counter."""
        self.timestamps.clear()