"""
Asynchronous frame processing pipeline for high FPS handling.
"""

import threading
import queue
import time
from typing import Optional, Callable, Any
import cv2
import numpy as np


class AsyncFrameProcessor:
    """Asynchronous frame processor that decouples frame capture from processing."""
    
    def __init__(self, process_func: Callable, max_queue_size: int = 5, num_workers: int = 2):
        """
        Initialize async processor.
        
        Args:
            process_func: Function to process frames (should accept frame and return result)
            max_queue_size: Maximum frames in processing queue
            num_workers: Number of worker threads for processing
        """
        self.process_func = process_func
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Queues
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        
        # Worker threads
        self.workers = []
        self.running = False
        
        # Statistics
        self.frames_queued = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = time.time()
        
    def start(self):
        """Start the async processing workers."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            self.workers.append(worker)
            worker.start()
        
        print(f"[ASYNC] Started {self.num_workers} processing workers")
    
    def stop(self):
        """Stop the async processing."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2)
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        print(f"[ASYNC] Stopped processing workers")
    
    def submit_frame(self, frame: np.ndarray, frame_id: int = None) -> bool:
        """
        Submit frame for processing.
        
        Args:
            frame: Frame to process
            frame_id: Optional frame identifier
            
        Returns:
            bool: True if frame was queued, False if dropped
        """
        try:
            # Non-blocking queue put
            self.input_queue.put((frame.copy(), frame_id, time.time()), block=False)
            self.frames_queued += 1
            return True
        except queue.Full:
            # Drop frame if queue is full
            self.frames_dropped += 1
            return False
    
    def get_result(self, timeout: float = 0.01) -> Optional[Any]:
        """
        Get processing result if available.
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            Processing result or None if none available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop."""
        while self.running:
            try:
                # Get frame from input queue
                try:
                    frame, frame_id, submit_time = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process frame
                start_time = time.time()
                result = self.process_func(frame)
                processing_time = time.time() - start_time
                
                # Put result in output queue
                result_data = {
                    'result': result,
                    'frame_id': frame_id,
                    'processing_time': processing_time,
                    'queue_delay': start_time - submit_time,
                    'worker_id': worker_id
                }
                
                self.result_queue.put(result_data)
                self.frames_processed += 1
                
                # Mark task as done
                self.input_queue.task_done()
                
            except Exception as e:
                print(f"[ASYNC-{worker_id}] Error: {e}")
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        elapsed = time.time() - self.start_time
        return {
            "frames_queued": self.frames_queued,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "input_queue_size": self.input_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "processing_fps": self.frames_processed / elapsed if elapsed > 0 else 0,
            "drop_rate": self.frames_dropped / max(1, self.frames_queued + self.frames_dropped)
        }


class FrameBuffer:
    """Lightweight frame buffer for smooth capture."""
    
    def __init__(self, buffer_size: int = 10):
        """Initialize frame buffer."""
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.dropped_frames = 0
        
    def put_frame(self, frame: np.ndarray) -> bool:
        """
        Add frame to buffer.
        
        Returns:
            bool: True if frame was added, False if dropped
        """
        try:
            self.buffer.put(frame.copy(), block=False)
            return True
        except queue.Full:
            # Remove oldest frame and add new one
            try:
                self.buffer.get_nowait()
                self.buffer.put(frame.copy(), block=False)
                self.dropped_frames += 1
                return True
            except queue.Empty:
                return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get frame from buffer."""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.buffer.empty()