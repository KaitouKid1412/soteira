#!/usr/bin/env python3
"""
Real-time video gating system for intelligent frame processing.
Implements 3-gate pipeline: Motion -> Scene-change -> Object-entry
"""

import argparse
import signal
import sys
import time
import csv
import os
from datetime import datetime
from pathlib import Path
import threading
import queue

import cv2
import numpy as np
import torch
import platform

# macOS OpenCV GUI initialization
if platform.system() == "Darwin":  # macOS
    try:
        # Initialize GUI backend early
        cv2.startWindowThread()
    except:
        pass

from gates.motion import MotionGate
from gates.scene import SceneGate
from gates.objects import ObjectGate
from llm_sink import LLMSink
from utils.timing import FPSCounter
from utils.io import safe_mkdir, safe_write_image, get_async_saver, shutdown_async_saver
from utils.image_quality import is_image_quality_good
from utils.frame_buffer import FrameBuffer
from utils.performance import FramePerformanceTracker
from utils.video_streamer import VideoStreamer
from utils.frame_sampler import AdaptiveFrameSampler, MotionBasedSampler
from utils.web_display import WebVideoDisplay


class VideoGatingSystem:
    def __init__(self, args):
        self.args = args
        self.save_dir = Path(args.save_dir)
        safe_mkdir(self.save_dir)
        
        # Initialize CSV logging
        self.csv_path = self.save_dir / "events_log.csv"
        self.init_csv_log()
        
        # Initialize gates
        self.motion_gate = MotionGate(threshold=args.motion_thresh)
        self.scene_gate = SceneGate(
            hist_threshold=args.scene_hist,
            ssim_threshold=args.scene_ssim,
            min_interval_ms=100  # Throttle scene detection to max 10 FPS
        )
        
        # Device selection for object detection
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"Using Apple Metal (MPS) for object detection")
        else:
            device = "cpu"
            print(f"Using CPU for object detection")
            
        self.object_gate = ObjectGate(
            device=device,
            confirm_hits=args.confirm_hits,
            max_age_seconds=args.max_age_seconds,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            min_area_frac=args.min_area_frac,
            classes=self.parse_classes(args.classes)
        )
        
        # LLM sink with intelligent prompt transformation and parallel processing
        # Will be updated with web display after web display is created
        self.llm_sink = LLMSink(
            user_query=args.prompt,
            openai_api_key=args.openai_api_key,
            dry_run=args.dry_run,
            num_workers=args.llm_workers,
            similarity_threshold=args.similarity_threshold,
            debug_mode=getattr(args, 'debug', False),
            mode=getattr(args, 'mode', 'summary'),
            streaming_mode=getattr(args, 'streaming_mode', False),
            gemini_api_key=getattr(args, 'gemini_api_key', None)
        )
        
        # Timing and display
        self.fps_counter = FPSCounter()
        self.frame_count = 0  # Frames actually processed
        self._total_frames_seen = 0  # Total frames from video (including dropped)
        self.running = True
        
        # Web-based display system
        self.web_display = None
        
        # Display control: web display enabled by default, unless --no-display is used
        if getattr(args, 'no_display', False):
            self.enable_web_display = False
        else:
            self.enable_web_display = True
        
        # Thread-safe logging
        self.log_lock = threading.Lock()
        
        # Gate trigger counters
        self.gate_triggers = {
            'motion': 0,
            'scene': 0,
            'object': 0
        }
        
        # Debug mode
        self.debug_mode = getattr(args, 'debug', False)
        
        # Frame buffer for quality capture
        self.frame_buffer = FrameBuffer(max_pending=10, capture_timeout=3.0)
        
        # Performance tracking
        self.perf_tracker = FramePerformanceTracker(save_dir=str(self.save_dir))
        
        # Async file saver for performance optimization
        self.async_saver = get_async_saver()
        
        # High FPS optimization
        if args.enable_60fps_mode:
            self.frame_sampler = AdaptiveFrameSampler(
                target_fps=args.target_fps,
                min_fps=args.min_fps, 
                max_fps=args.max_fps
            )
            self.enable_sampling = True
            print(f"[60FPS] Enabled adaptive sampling: target {args.target_fps} FPS")
        else:
            self.frame_sampler = None
            self.enable_sampling = False
        
        # Graceful shutdown state
        self.shutdown_requested = False
        self.shutdown_reason = ""
        
        # Event display state
        self.scene_flash_until = 0
        self.object_flash_until = 0
        self.object_flash_id = None
        
    def parse_classes(self, classes_str):
        if not classes_str:
            return None
        return [int(x.strip()) for x in classes_str.split(',')]
    
    def safe_print(self, message):
        """Thread-safe print function to fix log spacing issues."""
        with self.log_lock:
            print(message)
    
    def init_csv_log(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ts_iso', 'event_type', 'path', 'track_id_or_blank', 'd_hist', 'ssim', 'changed_ratio'])
    
    def log_event(self, event_type, path, track_id="", d_hist="", ssim="", changed_ratio=""):
        ts_iso = datetime.now().isoformat()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts_iso, event_type, path, track_id, d_hist, ssim, changed_ratio])
        print(f"[{ts_iso[:19]}] {event_type}: {path} (track_id={track_id})")
    
    def save_event_frame(self, frame, event_type, track_id=None):
        # Check image quality first
        is_good_quality, quality_metrics = is_image_quality_good(
            frame,
            blur_threshold=self.args.blur_threshold,
            min_brightness=self.args.min_brightness,
            max_brightness=self.args.max_brightness,
            min_contrast=self.args.min_contrast
        )
        
        if not is_good_quality:
            print(f"[QUALITY] Skipping {event_type} - poor quality: blur={quality_metrics['blur_score']:.1f}, brightness={quality_metrics['brightness']:.1f}, contrast={quality_metrics['contrast']:.1f}")
            return None
        
        ts = int(time.time() * 1000)
        if track_id is not None:
            filename = f"{event_type}_{track_id}_{ts}.jpg"
        else:
            filename = f"{event_type}_{ts}.jpg"
        
        path = self.save_dir / filename
        safe_write_image(str(path), frame)
        return str(path)
    
    def save_buffered_frame(self, frame, event_type, track_id=None, quality_metrics=None):
        """Save a frame that was selected from the buffer."""
        ts = int(time.time() * 1000)
        if track_id is not None:
            filename = f"{event_type}_{track_id}_{ts}.jpg"
        else:
            filename = f"{event_type}_{ts}.jpg"
        
        path = self.save_dir / filename
        
        # Use async saving for better performance
        def save_callback(success):
            if success and quality_metrics:
                print(f"[BUFFER] Saved {event_type} frame: blur={quality_metrics['blur_score']:.1f}, brightness={quality_metrics['brightness']:.1f}, contrast={quality_metrics['contrast']:.1f}")
            elif not success:
                print(f"[BUFFER] Failed to save {event_type} frame: {path}")
        
        success = self.async_saver.save_image_async(str(path), frame, callback=save_callback)
        
        # Return path immediately (async save will happen in background)
        return str(path) if success else None
    
    def draw_overlays(self, frame, motion_ratio, d_hist, ssim, detections):
        h, w = frame.shape[:2]
        
        # Status text
        status_y = 30
        cv2.putText(frame, f"Motion: {motion_ratio:.3f}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"dHist: {d_hist:.3f}", (10, status_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"SSIM: {ssim:.3f}", (10, status_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps_counter.get_fps():.1f}", (10, status_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {self.frame_buffer.get_pending_count()}", (10, status_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Detection boxes
        for det in detections:
            box = det['box']
            track_id = det.get('track_id', -1)
            conf = det['conf']
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Event flashes
        now = time.time()
        if now < self.scene_flash_until:
            cv2.putText(frame, "SCENE CHANGE", (w//2 - 100, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        if now < self.object_flash_until and self.object_flash_id:
            cv2.putText(frame, f"OBJECT ENTERED (id={self.object_flash_id})", 
                       (w//2 - 150, h//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    def process_frame(self, frame):
        self._total_frames_seen += 1  # Count every frame that enters processing
        self.frame_count += 1  # Will be reset if frame is dropped
        
        # High FPS sampling check
        if self.enable_sampling and not self.frame_sampler.should_process_frame():
            self.frame_count -= 1  # Revert since this frame won't be processed
            # Get specific reason for frame drop
            drop_reason = self.frame_sampler.get_last_drop_reason()
            dropped_count = self._total_frames_seen - self.frame_count
            if dropped_count % 30 == 0:  # Log every 30th drop to avoid spam
                print(f"[FRAME DROP] Total dropped: {dropped_count}, Reason: {drop_reason}")
            
            # Still update display even for dropped frames (for smooth preview)
            if self.enable_web_display:
                display_frame = frame.copy()
                # Draw minimal overlay for dropped frames
                cv2.putText(display_frame, "FRAME DROPPED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"FPS: {self.fps_counter.get_fps():.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                self.update_display_frame(display_frame)
            return  # Skip this frame
        
        frame_start_time = time.time()
        
        # Resize frame - use smaller size for 60fps mode
        if self.enable_sampling:
            frame_small = cv2.resize(frame, (480, 270))  # Smaller for speed
        else:
            frame_small = cv2.resize(frame, (640, 360))
        
        # ALWAYS update display first (regardless of motion detection)
        if self.enable_web_display:
            display_frame = frame.copy()
            # We'll add overlays after processing, but show frame immediately for responsiveness
            self.update_display_frame(display_frame)
            if self.debug_mode and self.frame_count == 1:
                print("[DEBUG] First frame sent to display immediately")
        
        # Detailed timing breakdown for bottleneck analysis
        timing_breakdown = {}
        last_time = frame_start_time
        
        current_time = time.time()
        timing_breakdown['resize'] = (current_time - last_time) * 1000
        last_time = current_time
        
        # Start performance tracking for this frame
        self.perf_tracker.start_frame(self.frame_count)
        
        # Process frame buffer - handle any completed captures
        buffer_start = time.time()
        completed_captures = self.frame_buffer.process_frame(frame)
        buffer_time = (time.time() - buffer_start) * 1000
        for capture in completed_captures:
            best_frame, quality_metrics = capture.get_best_frame()
            if best_frame is not None:
                # Save the best quality frame
                save_start = time.time()
                path = self.save_buffered_frame(best_frame, capture.event_type, capture.track_id, quality_metrics)
                save_time = (time.time() - save_start) * 1000
                if path:
                    self.log_event(capture.event_type + "_change" if capture.event_type == "scene" else "object_entered", 
                                 path, track_id=capture.track_id or "", 
                                 changed_ratio=self.current_changed_ratio)
                    
                    # Enqueue to LLM (dry-run is handled inside LLM sink)
                    # Pass frame data and motion info for smart prioritization
                    llm_queued = self.llm_sink.enqueue_to_llm(
                        self.args.prompt, path, best_frame, 
                        motion_ratio=getattr(self, 'current_changed_ratio', 0.0)
                    )
                    
                    # Record LLM event (only count as sent if actually queued)
                    self.perf_tracker.record_llm_event(llm_queued, self.llm_sink.get_queue_size())
                    
                    # Flash event
                    if capture.event_type == "scene":
                        self.scene_flash_until = time.time() + 1.0
                    else:
                        self.object_flash_until = time.time() + 1.0
                        self.object_flash_id = capture.track_id
            else:
                print(f"[BUFFER] No good quality frame found for {capture.event_type}")
        
        # Store current changed ratio for logging
        self.current_changed_ratio = 0.0
        
        # Gate controls - check individual skip flags or global bypass
        skip_motion = getattr(self.args, 'skip_motion', False) or getattr(self.args, 'bypass_gates', False)
        skip_scene = getattr(self.args, 'skip_scene', False) or getattr(self.args, 'bypass_gates', False) 
        
        # Gate 1: Motion detection (or skip if requested)
        if skip_motion:
            changed_ratio = 1.0  # Force motion spike
            motion_spike = True
            self.gate_triggers['motion'] += 1
            if self.debug_mode:
                print(f"[SKIP] Motion gate bypassed - processing frame {self.frame_count}")
        else:
            with self.perf_tracker.time_gate('motion'):
                changed_ratio, motion_mask = self.motion_gate.apply_motion(frame_small)
                motion_spike = changed_ratio > self.args.motion_thresh
                if motion_spike:
                    self.gate_triggers['motion'] += 1
        
        self.current_changed_ratio = changed_ratio
        self.perf_tracker.record_motion_result(motion_spike)
        
        # Gate 2: Scene change detection (or skip if requested)
        d_hist, ssim = 0.0, 1.0
        scene_changed = False
        
        if skip_scene:
            if motion_spike:  # Only if motion was detected (or motion was skipped)
                scene_changed = True
                d_hist, ssim = 1.0, 0.0  # Force scene change values
                self.gate_triggers['scene'] += 1
                if self.debug_mode:
                    print(f"[SKIP] Scene gate bypassed - processing frame {self.frame_count}")
        else:
            if motion_spike:
                with self.perf_tracker.time_gate('scene'):
                    scene_changed, d_hist, ssim = self.scene_gate.check_scene_change(frame_small)
                    if scene_changed:
                        self.gate_triggers['scene'] += 1
        
        self.perf_tracker.record_scene_result(scene_changed)
        
        if scene_changed:
            # Request buffered capture instead of immediate save
            self.frame_buffer.request_capture(
                "scene", 
                frames_to_capture=self.args.buffer_frames,
                blur_threshold=self.args.blur_threshold,
                min_brightness=self.args.min_brightness,
                max_brightness=self.args.max_brightness,
                min_contrast=self.args.min_contrast,
                disable_quality_filter=self.args.disable_quality_filter,
                stop_on_good_frame=self.args.stop_on_good_frame
            )
        
        # Gate 3: Object detection (or skip if requested)
        detections = []
        new_tracks = []
        skip_object = getattr(self.args, 'skip_object', False) or getattr(self.args, 'bypass_gates', False)
        
        if skip_object:
            # Skip object detection entirely - no new tracks will be created
            if self.debug_mode:
                print(f"[SKIP] Object gate bypassed - no object detection for frame {self.frame_count}")
        else:
            should_detect = (
                motion_spike or
                scene_changed
            )
            
            if should_detect:
                self.gate_triggers['object'] += 1
                with self.perf_tracker.time_gate('object'):
                    detections, new_tracks = self.object_gate.process_frame(frame_small, frame)
        
        self.perf_tracker.record_object_result(len(detections))
        
        # Handle new object entries
        for track in new_tracks:
            track_id = track['id']
            
            # Request buffered capture for new object
            self.frame_buffer.request_capture(
                "enter", 
                track_id=track_id,
                frames_to_capture=self.args.buffer_frames,
                blur_threshold=self.args.blur_threshold,
                min_brightness=self.args.min_brightness,
                max_brightness=self.args.max_brightness,
                min_contrast=self.args.min_contrast,
                disable_quality_filter=self.args.disable_quality_filter,
                stop_on_good_frame=self.args.stop_on_good_frame
            )
        
        # Update display with overlays (frame was already sent above for immediate response)
        if self.enable_web_display:
            # Create display frame with full overlays
            display_frame = frame.copy()
            self.draw_overlays(display_frame, changed_ratio, d_hist, ssim, detections)
            self.update_display_frame(display_frame)
        
        # Check if graceful shutdown was requested
        if self.shutdown_requested:
            self.perform_graceful_shutdown()
            return  # Exit main loop
        
        self.fps_counter.tick()
        
        # Add comprehensive frame statistics and timing breakdown (every 60 processed frames)
        if self.debug_mode and self.frame_count % 60 == 0:
            total_time = time.time() - frame_start_time
            processing_fps = self.fps_counter.get_fps()
            
            # Calculate drop statistics
            if hasattr(self, '_total_frames_seen'):
                drop_rate = ((self._total_frames_seen - self.frame_count) / self._total_frames_seen) * 100
                print(f"[FRAME STATS] Processed: {self.frame_count}, Seen: {self._total_frames_seen}, "
                      f"Dropped: {self._total_frames_seen - self.frame_count} ({drop_rate:.1f}%), "
                      f"Processing: {processing_fps:.1f} FPS, Frame time: {total_time*1000:.1f}ms")
                      
                # Show buffer timing if significant
                if 'buffer_time' in locals() and buffer_time > 1.0:
                    print(f"[BUFFER TIMING] Frame buffer processing: {buffer_time:.1f}ms")
                if 'save_time' in locals() and save_time > 1.0:
                    print(f"[SAVE TIMING] Frame saving: {save_time:.1f}ms")
                    
                # Show async save queue status
                save_stats = self.async_saver.get_stats()
                if save_stats['queue_size'] > 0:
                    print(f"[ASYNC_SAVE] Queue: {save_stats['queue_size']}/50, "
                          f"Completed: {save_stats['saves_completed']}, "
                          f"Failed: {save_stats['saves_failed']}")
            else:
                print(f"[TIMING] Frame {self.frame_count}: {total_time*1000:.1f}ms processing, {processing_fps:.1f} FPS effective")
        
        # Record sampling performance
        if self.enable_sampling:
            processing_time = time.time() - frame_start_time
            self.frame_sampler.record_processing_time(processing_time)
        
        # End performance tracking for this frame
        self.perf_tracker.end_frame()
    
    def start_web_display(self):
        """Start the web-based display system."""
        if not self.enable_web_display:
            return
        
        # Start web-based display
        port = getattr(self.args, 'web_port', 8888)
        # Get source FPS for real-time display
        source_fps = getattr(self, 'source_fps', 30.0)
        # Override with user-specified display FPS if provided
        display_fps = getattr(self.args, 'display_fps', None)
        if display_fps is None:
            display_fps = source_fps
        
        self.web_display = WebVideoDisplay(
            port=port, 
            target_fps=display_fps, 
            user_prompt=self.args.prompt,
            mode=getattr(self.args, 'mode', 'summary')
        )
        if self.web_display.start():
            print("[WEB_DISPLAY] Started successfully")
            print(f"[WEB_DISPLAY] View video at: http://localhost:{port}")
            print(f"[WEB_DISPLAY] Display FPS: {display_fps:.1f} (Source: {source_fps:.1f})")
            
            # Connect web display to LLM sink for notifications
            self.llm_sink.web_display = self.web_display
            
            return True
        else:
            print("[WEB_DISPLAY] Failed to start web server")
            self.enable_web_display = False
            return False
    
    def _display_loop(self):
        """Background thread for smooth video display."""
        import cv2
        import os
        import platform
        print("[DISPLAY] Display thread started")
        
        # macOS-specific fixes for OpenCV GUI
        if platform.system() == "Darwin":  # macOS
            try:
                # Try to set the GUI backend explicitly
                os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  
                # Force Qt backend if available
                try:
                    cv2.setUseOptimized(True)
                except:
                    pass
            except Exception as e:
                print(f"[DISPLAY] Warning: Could not set macOS OpenCV optimizations: {e}")
        
        # Initialize display window
        window_name = "Video Gating System - Real Time"
        window_created = False
        
        frame_count = 0
        frames_received = 0
        while self.running:  # Removed safety limit
            try:
                # Get latest frame (thread-safe)
                with self.display_lock:
                    frame_to_show = self.display_frame.copy() if self.display_frame is not None else None
                
                if frame_to_show is not None:
                    frames_received += 1
                    if frames_received == 1:
                        print("[DEBUG] Display thread received first frame")
                    elif frames_received % 30 == 0:
                        print(f"[DEBUG] Display thread received {frames_received} frames")
                    # Create window on first frame with macOS workarounds
                    if not window_created:
                        window_creation_success = False
                        
                        # Try multiple approaches for macOS compatibility
                        window_flags = [
                            cv2.WINDOW_NORMAL,
                            cv2.WINDOW_AUTOSIZE,
                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
                        ]
                        
                        for i, flag in enumerate(window_flags):
                            try:
                                print(f"[DISPLAY] Attempting window creation method {i+1}...")
                                cv2.namedWindow(window_name, flag)
                                
                                # Only resize for WINDOW_NORMAL
                                if flag == cv2.WINDOW_NORMAL:
                                    cv2.resizeWindow(window_name, 800, 600)
                                
                                # Test if window was actually created by trying to move it
                                cv2.moveWindow(window_name, 100, 100)
                                window_created = True
                                window_creation_success = True
                                print(f"[DISPLAY] Window created successfully using method {i+1}")
                                break
                                
                            except Exception as e:
                                print(f"[DISPLAY] Window creation method {i+1} failed: {e}")
                                # Try to destroy any partial window
                                try:
                                    cv2.destroyWindow(window_name)
                                except:
                                    pass
                        
                        if not window_creation_success:
                            print("[DISPLAY] All window creation methods failed. OpenCV GUI not available.")
                            print("[DISPLAY] This is common when running from Terminal on macOS.")
                            print("[DISPLAY] Try running from an IDE or with DISPLAY set.")
                            return
                    
                    try:
                        cv2.imshow(window_name, frame_to_show)
                        frame_count += 1
                    except Exception as e:
                        print(f"[DISPLAY] Failed to show frame: {e}")
                        break
                
                # Check for quit key (non-blocking)
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("[DISPLAY] User requested quit")
                        self.running = False
                        break
                except Exception as e:
                    print(f"[DISPLAY] waitKey error: {e}")
                    break
                    
                # Display at ~30 FPS
                time.sleep(0.033)
                
            except Exception as e:
                print(f"[DISPLAY] Error in display loop: {e}")
                break
        
        # Cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print(f"[DISPLAY] Display thread stopped after {frame_count} frames")
    
    def update_display_frame(self, frame):
        """Update the frame for web display."""
        if not self.enable_web_display or not self.web_display:
            return
        
        # Send to web display
        self.web_display.update_frame(frame)
    
    def run(self):
        # Setup video capture or streamer
        source = self.args.source
        cap = None
        using_streamer = False
        using_phone = False
        
        # Check if source is 'phone' - use HTTP phone capture
        if source.lower() == 'phone':
            from phone_http_capture import PhoneHTTPCapture
            cap = PhoneHTTPCapture()
            using_phone = True
            if not cap.isOpened():
                print("Error: Phone capture not available. Make sure video_server.py is running and phone is connected.")
                return
            print("Using phone stream via HTTP")
            # Disable web display for phone streams to reduce latency - the video server handles the web streaming
            print("[PHONE] Disabling web display for better performance - use video_server.py web interface")
            self.enable_web_display = False
        
        # Check if source is a video file with streaming enabled
        elif hasattr(self.args, 'stream_video') and self.args.stream_video and not source.isdigit():
            # Use video streamer for file input
            try:
                speed = getattr(self.args, 'stream_speed', 1.0)
                loop = getattr(self.args, 'stream_loop', True)
                cap = VideoStreamer(source, loop=loop, speed_multiplier=speed)
                cap.start_streaming()
                using_streamer = True
                print(f"Using video streamer for file: {source}")
            except Exception as e:
                print(f"Error creating video streamer: {e}")
                print("Falling back to regular video capture...")
                cap = None
        
        # Fallback to regular video capture
        if cap is None and not using_phone:
            if source.isdigit():
                source = int(source)
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return
        
        # Detect source FPS for real-time display
        if using_streamer:
            # For video streamer, get FPS from the streamer
            self.source_fps = getattr(cap, 'fps', 30.0)
        else:
            # For regular video capture, get FPS from OpenCV
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps > 0 and detected_fps < 120:  # Reasonable FPS range
                self.source_fps = detected_fps
            else:
                # Default FPS for webcams or invalid detection
                if isinstance(source, int):  # Webcam
                    self.source_fps = 30.0
                else:  # Video file with invalid FPS
                    self.source_fps = 25.0
        
        print(f"Detected source FPS: {self.source_fps:.1f}")
        
        print(f"Starting video gating system on source: {source}")
        print(f"Save directory: {self.save_dir}")
        print(f"Motion threshold: {self.args.motion_thresh}")
        print(f"Scene hist threshold: {self.args.scene_hist}")
        print(f"Scene SSIM threshold: {self.args.scene_ssim}")
        print(f"Track max age: {self.args.max_age_seconds}s")
        
        print(f"Web display enabled: {self.enable_web_display}")
        
        # Start web display system for real-time preview (only if enabled)
        if self.enable_web_display:
            print("Starting web display system...")
            self.start_web_display()
        else:
            print("Web display system disabled")
        
        # Main processing loop
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and source.startswith('rtsp'):
                        print("RTSP connection lost, attempting reconnect in 2s...")
                        time.sleep(2)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        continue
                    else:
                        print("End of video or read error")
                        # Mark video as ended for web display
                        if self.enable_web_display and self.web_display:
                            self.web_display.set_video_ended(True)
                            # Generate and set summary if in summary mode
                            if getattr(self.args, 'mode', 'summary') == 'summary':
                                summary = self.llm_sink.generate_synthesis_summary()
                                self.web_display.set_summary(summary)
                        break
                
                # Check if shutdown was requested before processing
                if not self.running:
                    print("Shutdown requested, stopping main loop")
                    break
                    
                self.process_frame(frame)
                
                # Check again after processing
                if not self.running:
                    print("Shutdown requested after frame processing")
                    break
                
                # Removed safety frame limit for full testing
                
        except KeyboardInterrupt:
            self.request_shutdown("Keyboard interrupt (Ctrl+C)")
        finally:
            # Perform graceful shutdown if requested
            if self.shutdown_requested:
                self.perform_graceful_shutdown()
            
            # Release video capture or streamer
            if using_streamer:
                cap.stop_streaming()
                # Show streaming stats
                stats = cap.get_stats()
                print(f"\n[STREAMER] Final stats: {stats['frames_streamed']} frames, "
                      f"{stats['actual_fps']:.1f} FPS avg")
            elif using_phone:
                # Show phone capture stats
                stats = cap.get_stats()
                print(f"\n[PHONE] Final stats: {stats.get('frames_read_http', 0)} received, "
                      f"{stats.get('frames_dropped', 0)} dropped, {stats.get('frames_read_http', 0)} processed")
            
            cap.release()
            
            # Generate summary before cleanup if in summary mode
            if getattr(self.args, 'mode', 'summary') == 'summary' and self.llm_sink.collected_responses:
                print("🔄 Generating final summary...")
                final_summary = self.llm_sink.generate_synthesis_summary()
                if self.enable_web_display and self.web_display:
                    self.web_display.set_summary(final_summary)
                    print("[WEB_DISPLAY] Summary set for web interface")
            
            # Cleanup web display system
            if self.enable_web_display and self.web_display:
                self.web_display.stop()
            self.llm_sink.shutdown()
            
            # Wait for async file saves to complete
            print("[SHUTDOWN] Waiting for async file saves to complete...")
            if not self.async_saver.wait_for_completion(timeout=10.0):
                print("[SHUTDOWN] Warning: Some file saves may not have completed")
            
            # Print async save stats
            save_stats = self.async_saver.get_stats()
            print(f"[ASYNC_SAVE] Stats: {save_stats['saves_completed']} completed, "
                  f"{save_stats['saves_failed']} failed, {save_stats['queue_full_drops']} dropped")
            
            # Shutdown async saver
            shutdown_async_saver()
            
            # Print gate trigger statistics
            print(f"\n{'='*60}")
            print(f"GATE TRIGGER STATISTICS")
            print(f"{'='*60}")
            print(f"Motion gate triggers: {self.gate_triggers['motion']}")
            print(f"Scene gate triggers: {self.gate_triggers['scene']}")  
            print(f"Object gate triggers: {self.gate_triggers['object']}")
            
            # Print LLM and deduplication stats
            llm_stats = self.llm_sink.get_stats()
            dedup_stats = llm_stats['similarity_dedup']
            cost_stats = llm_stats['cost_tracking']
            
            print(f"\nLLM PROCESSING & DEDUPLICATION ({self.llm_sink.mode.upper()} MODE):")
            print(f"Total images requested for LLM: {dedup_stats['total_requested']}")
            print(f"Images removed by deduplication: {dedup_stats['skipped_similar']}")
            print(f"Images sent to LLM: {dedup_stats['sent_to_llm']}")
            print(f"Deduplication efficiency: {dedup_stats['efficiency_percent']:.1f}%")
            print(f"Similarity threshold: {dedup_stats['threshold']}")
            
            # Mode-specific output
            if self.llm_sink.mode == "alert":
                print(f"🚨 Total alerts triggered: {self.llm_sink.alert_count}")
            elif self.llm_sink.mode == "summary":
                print(f"📝 Responses collected for summary: {len(self.llm_sink.collected_responses)}")
                # Generate and display synthesis
                if self.llm_sink.collected_responses:
                    print("🔄 Generating comprehensive answer...")
                    synthesis = self.llm_sink.generate_synthesis_summary()
                    print(f"\n{synthesis}")
                    
                    # Also set summary in web display if available
                    if self.enable_web_display and self.web_display:
                        self.web_display.set_summary(synthesis)
                else:
                    print("⚠️  No responses collected for summary")
                    # Set empty summary message in web display
                    if self.enable_web_display and self.web_display:
                        self.web_display.set_summary("No responses were collected for summary generation.")
            
            print(f"\nLLM COST TRACKING ({cost_stats['model']}):")
            print(f"Total tokens used: {cost_stats['total_tokens']:,}")
            print(f"  - Input tokens: {cost_stats['total_prompt_tokens']:,}")
            print(f"  - Output tokens: {cost_stats['total_completion_tokens']:,}")
            print(f"Total cost: ${cost_stats['total_cost_usd']:.4f}")
            if cost_stats['total_cost_usd'] > 0:
                avg_cost_per_image = cost_stats['total_cost_usd'] / max(dedup_stats['sent_to_llm'], 1)
                print(f"Average cost per image: ${avg_cost_per_image:.4f}")
            print(f"Pricing: ${cost_stats['input_cost_per_1m']}/1M input, ${cost_stats['output_cost_per_1m']}/1M output tokens")
            
            # Print performance statistics and save final data
            self.perf_tracker.print_statistics()
            self.perf_tracker.save_final_stats()
            
            # Print sampling statistics if enabled
            if self.enable_sampling:
                stats = self.frame_sampler.get_stats()
                print(f"\\n[60FPS] Sampling Stats:")
                print(f"  Frames seen: {stats['frames_seen']}")
                print(f"  Frames processed: {stats['frames_processed']}")
                print(f"  Skip rate: {stats['current_skip_rate']}")
                print(f"  Effective FPS: {stats['effective_fps']:.1f}")
                print(f"  Processing ratio: {stats['frames_processed']/stats['frames_seen']*100:.1f}%")
    
    def request_shutdown(self, reason="User request"):
        """Request graceful shutdown."""
        if not self.shutdown_requested:
            print(f"\n[SHUTDOWN] Graceful shutdown requested: {reason}")
            print(f"[SHUTDOWN] Finishing pending frame buffer operations...")
            self.shutdown_requested = True
            self.shutdown_reason = reason
    
    def perform_graceful_shutdown(self):
        """Perform graceful shutdown, finishing pending operations."""
        print(f"[SHUTDOWN] Processing {self.frame_buffer.get_pending_count()} pending captures...")
        
        # Set a reasonable timeout for pending captures
        shutdown_timeout = 10.0  # 10 seconds max
        shutdown_start = time.time()
        
        # Continue processing until all buffers are done or timeout
        while (self.frame_buffer.get_pending_count() > 0 and 
               time.time() - shutdown_start < shutdown_timeout):
            
            # Process any completed captures without new frame input
            completed_captures = self.frame_buffer.process_frame(None)
            
            for capture in completed_captures:
                best_frame, quality_metrics = capture.get_best_frame()
                if best_frame is not None:
                    # Save the best quality frame
                    path = self.save_buffered_frame(best_frame, capture.event_type, capture.track_id, quality_metrics)
                    if path:
                        self.log_event(capture.event_type + "_change" if capture.event_type == "scene" else "object_entered", 
                                     path, track_id=capture.track_id or "", 
                                     changed_ratio=getattr(self, 'current_changed_ratio', 0.0))
                        
                        # Enqueue to LLM (dry-run is handled inside LLM sink)
                        self.llm_sink.enqueue_to_llm(self.args.prompt, path, best_frame)
                else:
                    print(f"[SHUTDOWN] No good quality frame found for {capture.event_type}")
            
            # Small delay to avoid busy waiting
            if self.frame_buffer.get_pending_count() > 0:
                time.sleep(0.1)
        
        # Clear any remaining old captures
        remaining = self.frame_buffer.get_pending_count()
        if remaining > 0:
            print(f"[SHUTDOWN] Timeout reached, dropping {remaining} pending captures")
            self.frame_buffer.clear_old_captures()
        
        # Wait for LLM queue to finish
        llm_queue_size = self.llm_sink.get_queue_size()
        if llm_queue_size > 0:
            print(f"[SHUTDOWN] Waiting for {llm_queue_size} LLM requests to complete...")
            self.llm_sink.wait_for_completion(timeout=15)
        
        print(f"[SHUTDOWN] Graceful shutdown complete: {self.shutdown_reason}")
        self.running = False
    
    def shutdown(self):
        """Legacy shutdown method for compatibility."""
        self.request_shutdown("Legacy shutdown call")
        self.perform_graceful_shutdown()


# Global reference to system for signal handling
_global_system = None

def signal_handler(sig, frame):
    print(f"\nCtrl+C pressed - forcing exit...")
    import os
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time video gating system for intelligent frame processing"
    )
    
    parser.add_argument("--source", type=str, default="0",
                       help="Webcam index, file path, RTSP URL, or 'phone' for WebRTC phone stream")
    parser.add_argument("--stream-video", action="store_true",
                       help="Stream video file at real-time frame rate (for video files only)")
    parser.add_argument("--stream-speed", type=float, default=1.0,
                       help="Video streaming speed multiplier (default: 1.0)")
    parser.add_argument("--stream-loop", action="store_true", default=True,
                       help="Loop video when streaming (default: True)")
    parser.add_argument("--no-stream-loop", action="store_true",
                       help="Don't loop video when streaming")
    parser.add_argument("--save-dir", type=str, default="events",
                       help="Directory to save event frames")
    parser.add_argument("--prompt", type=str, 
                       default="fetch all the pictures of humans",
                       help="User query for AI analysis (e.g., 'find humans', 'check for cigarettes')")
    parser.add_argument("--openai-api-key", type=str, default=None,
                       help="OpenAI API key for GPT-4 Vision (uses stub if not provided)")
    parser.add_argument("--openai-model", type=str, default="gpt-4o",
                       help="OpenAI model to use (default: gpt-4o)")
    
    # Motion gate params
    parser.add_argument("--motion-thresh", type=float, default=0.02,
                       help="Motion threshold (fraction of pixels, default 0.02)")
    
    # Scene gate params
    parser.add_argument("--scene-hist", type=float, default=0.50,
                       help="Scene histogram threshold (default 0.50)")
    parser.add_argument("--scene-ssim", type=float, default=0.45,
                       help="Scene SSIM threshold (default 0.45)")
    
    # Object detection params
    parser.add_argument("--confirm-hits", type=int, default=2,
                       help="Confirm track after K consecutive hits (default 2)")
    parser.add_argument("--max-age-seconds", type=float, default=60.0,
                       help="Track max age in seconds since last seen (default 60.0)")
    parser.add_argument("--imgsz", type=int, default=416,
                       help="Detector input size (default 416)")
    parser.add_argument("--conf", type=float, default=0.4,
                       help="Detector confidence threshold (default 0.4)")
    parser.add_argument("--iou", type=float, default=0.7,
                       help="Detector IoU threshold (default 0.7)")
    parser.add_argument("--min-area-frac", type=float, default=0.01,
                       help="Ignore boxes smaller than this fraction (default 0.01)")
    parser.add_argument("--classes", type=str, default=None,
                       help="Comma-separated list of class IDs to keep (e.g., '0,2,5')")
    
    # Image quality control
    parser.add_argument("--blur-threshold", type=float, default=25.0,
                       help="Minimum blur score for sharp images (default 25.0)")
    parser.add_argument("--min-brightness", type=float, default=20.0,
                       help="Minimum image brightness (default 20.0)")
    parser.add_argument("--max-brightness", type=float, default=240.0,
                       help="Maximum image brightness (default 240.0)")
    parser.add_argument("--min-contrast", type=float, default=10.0,
                       help="Minimum image contrast (default 10.0)")
    parser.add_argument("--buffer-frames", type=int, default=10,
                       help="Number of frames to capture after event detection (default 10)")
    parser.add_argument("--disable-quality-filter", action="store_true",
                       help="Disable quality filtering (save best frame regardless of quality)")
    parser.add_argument("--stop-on-good-frame", action="store_true", default=True,
                       help="Stop capturing as soon as a good quality frame is found (default: True)")
    parser.add_argument("--capture-all-frames", action="store_true",
                       help="Capture all buffer frames and pick the best (opposite of --stop-on-good-frame)")
    
    # Display and control
    parser.add_argument("--web-port", type=int, default=8888,
                       help="Port for web display server (default: 8888)")
    parser.add_argument("--display-fps", type=float, default=None,
                       help="Force display FPS (overrides auto-detection, e.g. --display-fps 30)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable web display completely")
    parser.add_argument("--dry-run", action="store_true",
                       help="Do not call LLM, just log/save")
    parser.add_argument("--llm-workers", type=int, default=8,
                       help="Number of parallel LLM worker threads (default: 8, optimized for real-time)")
    parser.add_argument("--similarity-threshold", type=float, default=0.75,
                       help="Image similarity threshold for deduplication (0.75 = 75% similar = skip, optimized for speed)")
    parser.add_argument("--mode", type=str, choices=["summary", "alert", "realtime_description"], default="summary",
                       help="Processing mode: 'summary' (combine all responses), 'alert' (check each response for alerts), or 'realtime_description' (continuous scene descriptions for accessibility)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug logging")
    
    # High FPS optimization
    parser.add_argument("--enable-60fps-mode", action="store_true",
                       help="Enable optimizations for 60 FPS processing")
    parser.add_argument("--target-fps", type=float, default=30.0,
                       help="Target processing FPS for adaptive sampling (default: 30.0)")
    parser.add_argument("--min-fps", type=float, default=10.0,
                       help="Minimum processing FPS (default: 10.0)")
    parser.add_argument("--max-fps", type=float, default=60.0,
                       help="Maximum processing FPS (default: 60.0)")
    parser.add_argument("--bypass-gates", action="store_true",
                       help="Bypass all gates (motion/scene/object) - process every frame")
    parser.add_argument("--skip-motion", action="store_true", 
                       help="Skip motion detection gate - process all frames for scene/object detection")
    parser.add_argument("--skip-scene", action="store_true",
                       help="Skip scene change detection gate - run object detection without scene requirement") 
    parser.add_argument("--skip-object", action="store_true",
                       help="Skip object detection gate - only save scene change frames")
    
    # Streaming and LLM options
    parser.add_argument("--streaming-mode", action="store_true",
                       help="Enable streaming mode with Gemini Flash 2.5 for real-time token delivery")
    parser.add_argument("--gemini-api-key", type=str,
                       help="Google Gemini API key for streaming mode")
    
    args = parser.parse_args()
    
    # Handle conflicting arguments
    if args.capture_all_frames:
        args.stop_on_good_frame = False
    
    # Handle stream loop arguments
    if args.no_stream_loop:
        args.stream_loop = False
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run system
    global _global_system
    system = VideoGatingSystem(args)
    _global_system = system  # Store global reference for signal handler
    
    try:
        system.run()
    finally:
        _global_system = None  # Clear global reference


if __name__ == "__main__":
    main()