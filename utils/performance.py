"""
Performance tracking utilities for video gating system.
"""

import time
import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class FramePerformanceTracker:
    """Track performance metrics for frame processing."""
    
    def __init__(self, save_dir: str = "events"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Performance data storage
        self.frame_times = []  # List of frame processing time records
        self.start_time = time.time()
        
        # Current frame tracking
        self.current_frame_start = None
        self.current_frame_number = 0
        
        # Gate-specific timings
        self.gate_times = {
            'motion': [],
            'scene': [],
            'object': [],
            'total': []
        }
        
        # LLM processing stats
        self.llm_queue_count = 0
        self.frames_sent_to_llm = 0
        
        # CSV file for detailed frame timings
        self.csv_path = self.save_dir / "frame_performance.csv"
        self.init_csv()
    
    def init_csv(self):
        """Initialize CSV file for frame performance logging."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_number', 'timestamp_iso', 'total_time_ms', 
                'motion_time_ms', 'scene_time_ms', 'object_time_ms',
                'motion_detected', 'scene_changed', 'objects_detected',
                'sent_to_llm', 'llm_queue_size'
            ])
    
    def start_frame(self, frame_number: int):
        """Mark the start of frame processing."""
        self.current_frame_start = time.time()
        self.current_frame_number = frame_number
        
        # Reset gate timings for this frame
        self.current_gate_times = {
            'motion': 0,
            'scene': 0, 
            'object': 0
        }
        self.current_events = {
            'motion_detected': False,
            'scene_changed': False,
            'objects_detected': 0,
            'sent_to_llm': False
        }
    
    def time_gate(self, gate_name: str):
        """Context manager for timing individual gates."""
        return GateTimer(self, gate_name)
    
    def record_gate_time(self, gate_name: str, duration_ms: float):
        """Record timing for a specific gate."""
        if gate_name in self.current_gate_times:
            self.current_gate_times[gate_name] = duration_ms
    
    def record_motion_result(self, motion_detected: bool):
        """Record motion detection result."""
        self.current_events['motion_detected'] = motion_detected
    
    def record_scene_result(self, scene_changed: bool):
        """Record scene change result."""
        self.current_events['scene_changed'] = scene_changed
    
    def record_object_result(self, objects_detected: int):
        """Record object detection result."""
        self.current_events['objects_detected'] = objects_detected
    
    def record_llm_event(self, sent_to_llm: bool, llm_queue_size: int = 0):
        """Record LLM processing event."""
        if sent_to_llm:
            self.current_events['sent_to_llm'] = True
            self.frames_sent_to_llm += 1
            self.llm_queue_count = llm_queue_size
    
    def end_frame(self):
        """Mark the end of frame processing and log results."""
        if self.current_frame_start is None:
            return
        
        # Calculate total frame time
        total_time = (time.time() - self.current_frame_start) * 1000  # Convert to ms
        
        # Store timing data
        frame_record = {
            'frame_number': self.current_frame_number,
            'timestamp': datetime.now().isoformat(),
            'total_time_ms': total_time,
            'motion_time_ms': self.current_gate_times.get('motion', 0),
            'scene_time_ms': self.current_gate_times.get('scene', 0),
            'object_time_ms': self.current_gate_times.get('object', 0),
            **self.current_events,
            'llm_queue_size': self.llm_queue_count
        }
        
        self.frame_times.append(frame_record)
        
        # Log to CSV
        self.log_frame_to_csv(frame_record)
        
        # Store gate-specific times for statistics
        self.gate_times['total'].append(total_time)
        for gate, time_ms in self.current_gate_times.items():
            if time_ms > 0:  # Only record if gate was actually used
                self.gate_times[gate].append(time_ms)
        
        # Reset current frame tracking
        self.current_frame_start = None
    
    def log_frame_to_csv(self, record: Dict):
        """Log frame performance record to CSV."""
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record['frame_number'],
                    record['timestamp'],
                    f"{record['total_time_ms']:.2f}",
                    f"{record['motion_time_ms']:.2f}",
                    f"{record['scene_time_ms']:.2f}",
                    f"{record['object_time_ms']:.2f}",
                    record['motion_detected'],
                    record['scene_changed'],
                    record['objects_detected'],
                    record['sent_to_llm'],
                    record['llm_queue_size']
                ])
        except Exception as e:
            print(f"[PERF] Error logging to CSV: {e}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive performance statistics."""
        if not self.frame_times:
            return {
                'total_frames': 0,
                'total_runtime_seconds': 0,
                'frames_sent_to_llm': 0,
                'avg_fps': 0
            }
        
        total_runtime = time.time() - self.start_time
        total_frames = len(self.frame_times)
        
        stats = {
            'total_frames': total_frames,
            'total_runtime_seconds': total_runtime,
            'frames_sent_to_llm': self.frames_sent_to_llm,
            'avg_fps': total_frames / total_runtime if total_runtime > 0 else 0
        }
        
        # Calculate timing statistics for each component
        for component, times in self.gate_times.items():
            if times:
                stats[f'{component}_min_ms'] = min(times)
                stats[f'{component}_max_ms'] = max(times)
                stats[f'{component}_mean_ms'] = statistics.mean(times)
                stats[f'{component}_median_ms'] = statistics.median(times)
                stats[f'{component}_frames_processed'] = len(times)
            else:
                stats[f'{component}_min_ms'] = 0
                stats[f'{component}_max_ms'] = 0
                stats[f'{component}_mean_ms'] = 0
                stats[f'{component}_median_ms'] = 0
                stats[f'{component}_frames_processed'] = 0
        
        return stats
    
    def print_statistics(self):
        """Print comprehensive performance statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE STATISTICS")
        print(f"{'='*60}")
        
        print(f"Runtime: {stats['total_runtime_seconds']:.1f}s")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Images sent to LLM: {stats['frames_sent_to_llm']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        
        print(f"\nFRAME PROCESSING TIMES:")
        if stats.get('total_frames_processed', 0) > 0:
            print(f"  Total (full pipeline):")
            print(f"    Min:    {stats.get('total_min_ms', 0):.2f}ms")
            print(f"    Max:    {stats.get('total_max_ms', 0):.2f}ms") 
            print(f"    Mean:   {stats.get('total_mean_ms', 0):.2f}ms")
            print(f"    Median: {stats.get('total_median_ms', 0):.2f}ms")
            print(f"    Frames: {stats.get('total_frames_processed', 0)}")
        else:
            print(f"  Total (full pipeline): No data available")
        
        for gate in ['motion', 'scene', 'object']:
            if stats[f'{gate}_frames_processed'] > 0:
                print(f"  {gate.capitalize()} gate:")
                print(f"    Min:    {stats[f'{gate}_min_ms']:.2f}ms")
                print(f"    Max:    {stats[f'{gate}_max_ms']:.2f}ms")
                print(f"    Mean:   {stats[f'{gate}_mean_ms']:.2f}ms")
                print(f"    Median: {stats[f'{gate}_median_ms']:.2f}ms")
                print(f"    Frames: {stats[f'{gate}_frames_processed']}")
        
        print(f"\nPerformance log saved to: {self.csv_path}")
        print(f"{'='*60}")
    
    def save_final_stats(self):
        """Save final statistics to JSON file."""
        stats = self.get_statistics()
        
        stats_file = self.save_dir / "performance_summary.json"
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Performance summary saved to: {stats_file}")


class GateTimer:
    """Context manager for timing gate operations."""
    
    def __init__(self, tracker: FramePerformanceTracker, gate_name: str):
        self.tracker = tracker
        self.gate_name = gate_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.tracker.record_gate_time(self.gate_name, duration_ms)