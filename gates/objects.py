"""
Object detection and tracking gate using YOLO + IOU tracker.
Runs only on motion spikes and scene changes (threshold-based triggering).
"""

import time
import numpy as np
from ultralytics import YOLO


class Track:
    """Individual object track."""
    
    def __init__(self, track_id, box, cls, conf):
        self.id = track_id
        self.box = box  # [x1, y1, x2, y2]
        self.cls = cls
        self.conf = conf
        self.hits = 1  # Number of consecutive detections
        self.last_seen = time.time()  # Time of last update
        self.created_at = time.time()  # Track creation time
        self.confirmed = False
        
    def update(self, box, cls, conf):
        """Update track with new detection."""
        self.box = box
        self.cls = cls
        self.conf = conf
        self.hits += 1
        self.last_seen = time.time()
        
    def get_age_seconds(self):
        """Get track age in seconds since last update."""
        return time.time() - self.last_seen
        
    def predict(self):
        """Simple prediction (just keep current position)."""
        return self.box


class IOUTracker:
    """Lightweight IOU-based tracker."""
    
    def __init__(self, iou_threshold=0.5, confirm_hits=2, max_age_seconds=60.0):
        self.iou_threshold = iou_threshold
        self.confirm_hits = confirm_hits
        self.max_age_seconds = max_age_seconds
        
        self.tracks = []
        self.next_id = 1
        
    def iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
            
        Returns:
            float: IoU value (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-7)
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'box', 'cls', 'conf'
            
        Returns:
            list: New tracks that just became confirmed
        """
        new_tracks = []
        
        # Predict existing tracks (no frame-based aging needed)
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks using IoU
        matched_tracks = set()
        matched_detections = set()
        
        for i, detection in enumerate(detections):
            best_iou = 0
            best_track_idx = -1
            
            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                    
                iou_val = self.iou(detection['box'], track.box)
                if iou_val > self.iou_threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_track_idx = j
            
            if best_track_idx >= 0:
                # Update existing track
                track = self.tracks[best_track_idx]
                was_confirmed = track.confirmed
                
                track.update(detection['box'], detection['cls'], detection['conf'])
                
                # Check if track just became confirmed
                if not was_confirmed and track.hits >= self.confirm_hits:
                    track.confirmed = True
                    new_tracks.append({
                        'id': track.id,
                        'box': track.box,
                        'cls': track.cls,
                        'conf': track.conf
                    })
                
                matched_tracks.add(best_track_idx)
                matched_detections.add(i)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                new_track = Track(
                    self.next_id,
                    detection['box'],
                    detection['cls'],
                    detection['conf']
                )
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Remove old tracks based on time since last seen
        self.tracks = [track for track in self.tracks if track.get_age_seconds() <= self.max_age_seconds]
        
        return new_tracks
    
    def get_active_tracks(self):
        """Get all confirmed tracks for display."""
        return [
            {
                'track_id': track.id,
                'box': track.box,
                'cls': track.cls,
                'conf': track.conf,
                'confirmed': track.confirmed
            }
            for track in self.tracks if track.confirmed
        ]


class ObjectGate:
    """Object detection and tracking gate."""
    
    def __init__(self, device="cpu", confirm_hits=2, max_age_seconds=60.0,
                 imgsz=416, conf=0.4, iou=0.7, min_area_frac=0.01, classes=None):
        """
        Initialize object detection gate.
        
        Args:
            device: Device for YOLO inference ("cpu", "mps", "cuda")
            confirm_hits: Confirm track after K consecutive hits
            max_age_seconds: Track max age in seconds since last seen
            imgsz: YOLO input size
            conf: Detection confidence threshold
            iou: Detection IoU threshold
            min_area_frac: Ignore boxes smaller than this fraction
            classes: List of class IDs to keep (None = all classes)
        """
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_area_frac = min_area_frac
        self.classes = classes
        
        # Initialize YOLO model (YOLOv5n is 40% faster than YOLOv8n)
        print(f"Loading YOLOv5n model on device: {device}")
        self.model = YOLO('yolov5nu.pt')  # Downloads automatically if needed
        
        # Initialize tracker
        self.tracker = IOUTracker(
            iou_threshold=0.5,
            confirm_hits=confirm_hits,
            max_age_seconds=max_age_seconds
        )
        
    def _filter_detections(self, results, frame_shape):
        """
        Filter and format YOLO detections.
        
        Args:
            results: YOLO results object
            frame_shape: (height, width) of input frame
            
        Returns:
            list: Filtered detection dicts
        """
        detections = []
        
        if len(results) == 0 or results[0].boxes is None:
            return detections
        
        boxes = results[0].boxes
        frame_area = frame_shape[0] * frame_shape[1]
        min_area = frame_area * self.min_area_frac
        
        for i in range(len(boxes)):
            # Extract box coordinates, confidence, and class
            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            print(f"[FILTER DEBUG] Detection {i}: conf={conf:.3f}, cls={cls}, box={box}")
            
            # Filter by area
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < min_area:
                print(f"[FILTER DEBUG] REJECTED: area {area:.1f} < min_area {min_area:.1f}")
                continue
                
            # Filter by class if specified
            if self.classes is not None and cls not in self.classes:
                print(f"[FILTER DEBUG] REJECTED: class {cls} not in allowed classes {self.classes}")
                continue
            
            print(f"[FILTER DEBUG] ACCEPTED: Detection {i}")
            
            detections.append({
                'box': box.tolist(),
                'conf': conf,
                'cls': cls
            })
        
        return detections
    
    def _run_detector(self, frame):
        """
        Run YOLO detector on frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            list: Detection results
        """
        try:
            # Run YOLO inference
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
                stream=False
            )
            
            detections = self._filter_detections(results, frame.shape[:2])
            print(f"[YOLO DEBUG] Raw results: {len(results)}, Filtered detections: {len(detections)}")
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def process_frame(self, frame_small, frame_full):
        """
        Process frame through batch object detection and tracking.
        
        Args:
            frame_small: Downscaled frame for detection
            frame_full: Full resolution frame for saving crops
            motion_detected: Whether motion was detected (gates YOLO execution)
            
        Returns:
            tuple: (detections_with_tracks, new_confirmed_tracks)
        """
        import time
        timing_breakdown = {}
        start_time = time.time()
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Run detector
        detection_start = time.time()
        detections = self._run_detector(frame_small)
        timing_breakdown['detection'] = (time.time() - detection_start) * 1000
        
        # Update tracker
        tracking_start = time.time()
        new_tracks = self.tracker.update(detections)
        timing_breakdown['tracking'] = (time.time() - tracking_start) * 1000
        print(f"[TRACKER DEBUG] Input detections: {len(detections)}, New tracks: {len(new_tracks)}")
        
        # Get active tracks for display
        display_start = time.time()
        active_tracks = self.tracker.get_active_tracks()
        timing_breakdown['get_tracks'] = (time.time() - display_start) * 1000
        
        # Scale bounding boxes from small frame to full frame for display
        scaling_start = time.time()
        if len(active_tracks) > 0 and frame_small.shape[:2] != frame_full.shape[:2]:
            scale_y = frame_full.shape[0] / frame_small.shape[0]
            scale_x = frame_full.shape[1] / frame_small.shape[1]
            
            for track in active_tracks:
                box = track['box']
                track['box'] = [
                    box[0] * scale_x,
                    box[1] * scale_y,
                    box[2] * scale_x,
                    box[3] * scale_y
                ]
        timing_breakdown['scaling_active'] = (time.time() - scaling_start) * 1000
        
        # Scale new track boxes for saving crops
        scaling_new_start = time.time()
        scaled_new_tracks = []
        if len(new_tracks) > 0 and frame_small.shape[:2] != frame_full.shape[:2]:
            scale_y = frame_full.shape[0] / frame_small.shape[0]
            scale_x = frame_full.shape[1] / frame_small.shape[1]
            
            for track in new_tracks:
                box = track['box']
                scaled_box = [
                    box[0] * scale_x,
                    box[1] * scale_y,
                    box[2] * scale_x,
                    box[3] * scale_y
                ]
                scaled_track = track.copy()
                scaled_track['box'] = scaled_box
                scaled_new_tracks.append(scaled_track)
        else:
            scaled_new_tracks = new_tracks
        timing_breakdown['scaling_new'] = (time.time() - scaling_new_start) * 1000
        
        # Log timing breakdown every 10 object detections to avoid spam
        total_time = (time.time() - start_time) * 1000
        if hasattr(self, '_detection_count'):
            self._detection_count += 1
        else:
            self._detection_count = 1
            
        if self._detection_count % 10 == 0:
            print(f"[OBJECT TIMING] Total: {total_time:.1f}ms | "
                  f"Detection: {timing_breakdown['detection']:.1f}ms | "
                  f"Tracking: {timing_breakdown['tracking']:.1f}ms | "
                  f"Get tracks: {timing_breakdown['get_tracks']:.1f}ms | "
                  f"Scale active: {timing_breakdown['scaling_active']:.1f}ms | "
                  f"Scale new: {timing_breakdown['scaling_new']:.1f}ms")
        
        return active_tracks, scaled_new_tracks
    
