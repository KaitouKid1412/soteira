"""
I/O utilities for safe file operations.
"""

import os
import cv2
import json
import tempfile
import threading
import queue
import time
from pathlib import Path
from datetime import datetime


def safe_mkdir(path):
    """
    Safely create directory and all parent directories.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_write_image(filepath, image, quality=95):
    """
    Safely write image to file with atomic operation.
    
    Args:
        filepath: Output file path
        image: OpenCV image (numpy array)
        quality: JPEG quality (0-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        parent_dir = Path(filepath).parent
        safe_mkdir(parent_dir)
        
        # Write to temporary file first (keep same extension for OpenCV)
        file_path = Path(filepath)
        temp_path = str(file_path.parent / (file_path.stem + "_tmp" + file_path.suffix))
        
        # Set compression parameters
        if filepath.lower().endswith(('.jpg', '.jpeg')):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif filepath.lower().endswith('.png'):
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            params = []
        
        # Write image
        success = cv2.imwrite(temp_path, image, params)
        
        if success:
            # Atomic move to final location
            os.rename(temp_path, filepath)
            return True
        else:
            # Clean up temp file if write failed
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
            
    except Exception as e:
        print(f"Error writing image {filepath}: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def safe_write_json(filepath, data, indent=2):
    """
    Safely write JSON data to file with atomic operation.
    
    Args:
        filepath: Output file path
        data: Data to serialize as JSON
        indent: JSON indentation (None for compact)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        parent_dir = Path(filepath).parent
        safe_mkdir(parent_dir)
        
        # Write to temporary file first
        temp_path = filepath + ".tmp"
        
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        # Atomic move to final location
        os.rename(temp_path, filepath)
        return True
        
    except Exception as e:
        print(f"Error writing JSON {filepath}: {e}")
        # Clean up temp file if write failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def safe_read_json(filepath, default=None):
    """
    Safely read JSON data from file.
    
    Args:
        filepath: Input file path
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Data from JSON file or default value
    """
    try:
        if not os.path.exists(filepath):
            return default
            
        with open(filepath, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"Error reading JSON {filepath}: {e}")
        return default


def get_file_size(filepath):
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        int: File size in bytes, -1 if error
    """
    try:
        return os.path.getsize(filepath)
    except Exception:
        return -1


def get_file_age_seconds(filepath):
    """
    Get file age in seconds since last modification.
    
    Args:
        filepath: Path to file
        
    Returns:
        float: Age in seconds, -1 if error
    """
    try:
        import time
        mtime = os.path.getmtime(filepath)
        return time.time() - mtime
    except Exception:
        return -1


def cleanup_old_files(directory, max_age_seconds, pattern="*"):
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean
        max_age_seconds: Maximum file age before deletion
        pattern: File pattern to match (default: all files)
        
    Returns:
        int: Number of files deleted
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        deleted_count = 0
        current_time = datetime.now().timestamp()
        
        for filepath in directory.glob(pattern):
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {filepath}: {e}")
        
        return deleted_count
        
    except Exception as e:
        print(f"Error cleaning up directory {directory}: {e}")
        return 0


def get_directory_size(directory):
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        int: Total size in bytes
    """
    try:
        total_size = 0
        directory = Path(directory)
        
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                total_size += filepath.stat().st_size
        
        return total_size
        
    except Exception as e:
        print(f"Error calculating directory size {directory}: {e}")
        return 0


def format_bytes(num_bytes):
    """
    Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        str: Formatted string (e.g., "1.2 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def ensure_unique_filename(filepath):
    """
    Ensure filename is unique by adding counter if needed.
    
    Args:
        filepath: Desired file path
        
    Returns:
        str: Unique file path
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return str(filepath)
    
    # Extract parts
    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    
    # Find unique name
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        
        if not new_path.exists():
            return str(new_path)
        
        counter += 1


def create_temp_file(suffix=".tmp", prefix="temp_", dir=None):
    """
    Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file (None for system default)
        
    Returns:
        str: Path to temporary file
    """
    try:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        os.close(fd)  # Close file descriptor
        return path
    except Exception as e:
        print(f"Error creating temp file: {e}")
        return None


def copy_file(src, dst):
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if successful
    """
    try:
        import shutil
        
        # Ensure destination directory exists
        dst_path = Path(dst)
        safe_mkdir(dst_path.parent)
        
        shutil.copy2(src, dst)
        return True
        
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def move_file(src, dst):
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if successful
    """
    try:
        import shutil
        
        # Ensure destination directory exists
        dst_path = Path(dst)
        safe_mkdir(dst_path.parent)
        
        shutil.move(src, dst)
        return True
        
    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}")
        return False


class FileRotator:
    """Utility for rotating log files or image files."""
    
    def __init__(self, base_path, max_files=10):
        """
        Initialize file rotator.
        
        Args:
            base_path: Base file path (without number)
            max_files: Maximum number of files to keep
        """
        self.base_path = Path(base_path)
        self.max_files = max_files
    
    def get_next_path(self):
        """Get path for next file in rotation."""
        stem = self.base_path.stem
        suffix = self.base_path.suffix
        parent = self.base_path.parent
        
        # Clean up old files first
        self._cleanup_old_files()
        
        # Find next available number
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter:03d}{suffix}"
            if not new_path.exists():
                return str(new_path)
            counter += 1
    
    def _cleanup_old_files(self):
        """Remove excess files keeping only max_files."""
        stem = self.base_path.stem
        suffix = self.base_path.suffix
        parent = self.base_path.parent
        
        # Find all numbered files
        pattern = f"{stem}_*{suffix}"
        files = list(parent.glob(pattern))
        
        if len(files) >= self.max_files:
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest files
            num_to_remove = len(files) - self.max_files + 1
            for i in range(num_to_remove):
                try:
                    files[i].unlink()
                except Exception as e:
                    print(f"Error removing old file {files[i]}: {e}")


class AsyncFileSaver:
    """Asynchronous file saver to decouple I/O from main processing loop."""
    
    def __init__(self, max_queue_size=50, num_workers=2):
        """
        Initialize async file saver.
        
        Args:
            max_queue_size: Maximum pending saves in queue
            num_workers: Number of background worker threads
        """
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        self.num_workers = num_workers
        self.workers = []
        self.running = True
        self.stats = {
            'saves_requested': 0,
            'saves_completed': 0,
            'saves_failed': 0,
            'queue_full_drops': 0
        }
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def save_image_async(self, filepath, image, quality=95, callback=None):
        """
        Queue an image for asynchronous saving.
        
        Args:
            filepath: Output file path
            image: OpenCV image (numpy array) - will be copied
            quality: JPEG quality (0-100)
            callback: Optional function to call when save completes (success: bool)
            
        Returns:
            bool: True if queued successfully, False if queue full
        """
        try:
            # Copy image to avoid race conditions
            image_copy = image.copy()
            
            save_task = {
                'type': 'image',
                'filepath': filepath,
                'image': image_copy,
                'quality': quality,
                'callback': callback,
                'timestamp': time.time()
            }
            
            self.save_queue.put_nowait(save_task)
            self.stats['saves_requested'] += 1
            return True
            
        except queue.Full:
            self.stats['queue_full_drops'] += 1
            if callback:
                callback(False)  # Notify failure
            return False
    
    def save_json_async(self, filepath, data, indent=2, callback=None):
        """
        Queue JSON data for asynchronous saving.
        
        Args:
            filepath: Output file path
            data: Data to serialize as JSON
            indent: JSON indentation
            callback: Optional function to call when save completes (success: bool)
            
        Returns:
            bool: True if queued successfully, False if queue full
        """
        try:
            save_task = {
                'type': 'json',
                'filepath': filepath,
                'data': data,
                'indent': indent,
                'callback': callback,
                'timestamp': time.time()
            }
            
            self.save_queue.put_nowait(save_task)
            self.stats['saves_requested'] += 1
            return True
            
        except queue.Full:
            self.stats['queue_full_drops'] += 1
            if callback:
                callback(False)  # Notify failure
            return False
    
    def _worker_loop(self):
        """Main worker loop for processing save tasks."""
        while self.running:
            try:
                # Get task with timeout to allow shutdown
                task = self.save_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                # Process the save task
                success = self._process_save_task(task)
                
                # Update stats
                if success:
                    self.stats['saves_completed'] += 1
                else:
                    self.stats['saves_failed'] += 1
                
                # Call callback if provided
                if task['callback']:
                    task['callback'](success)
                
                self.save_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                print(f"[ASYNC_SAVE] Worker error: {e}")
                self.stats['saves_failed'] += 1
    
    def _process_save_task(self, task):
        """Process a single save task."""
        try:
            if task['type'] == 'image':
                return safe_write_image(
                    task['filepath'], 
                    task['image'], 
                    task['quality']
                )
            elif task['type'] == 'json':
                return safe_write_json(
                    task['filepath'], 
                    task['data'], 
                    task['indent']
                )
            else:
                print(f"[ASYNC_SAVE] Unknown task type: {task['type']}")
                return False
                
        except Exception as e:
            print(f"[ASYNC_SAVE] Error processing {task['type']} save: {e}")
            return False
    
    def get_stats(self):
        """Get save statistics."""
        stats = self.stats.copy()
        stats['queue_size'] = self.save_queue.qsize()
        stats['pending_ratio'] = self.save_queue.qsize() / self.save_queue.maxsize
        return stats
    
    def wait_for_completion(self, timeout=10.0):
        """
        Wait for all pending saves to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if all saves completed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.save_queue.empty():
                return True
            time.sleep(0.1)
        
        return False
    
    def shutdown(self, timeout=5.0):
        """
        Shutdown the async file saver.
        
        Args:
            timeout: Maximum time to wait for workers to finish
        """
        print(f"[ASYNC_SAVE] Shutting down with {self.save_queue.qsize()} pending saves...")
        
        # Stop accepting new tasks
        self.running = False
        
        # Send shutdown signals to workers
        for _ in self.workers:
            try:
                self.save_queue.put_nowait(None)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        # Final stats
        stats = self.get_stats()
        print(f"[ASYNC_SAVE] Final stats: {stats['saves_completed']} completed, "
              f"{stats['saves_failed']} failed, {stats['queue_full_drops']} dropped")


# Global async file saver instance
_global_async_saver = None

def get_async_saver():
    """Get global async file saver instance."""
    global _global_async_saver
    if _global_async_saver is None:
        _global_async_saver = AsyncFileSaver()
    return _global_async_saver

def shutdown_async_saver():
    """Shutdown global async file saver."""
    global _global_async_saver
    if _global_async_saver is not None:
        _global_async_saver.shutdown()
        _global_async_saver = None