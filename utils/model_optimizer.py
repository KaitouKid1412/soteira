"""
Model optimization utilities for faster object detection.
"""

import time
import torch
from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Tuple


class ModelBenchmark:
    """Benchmark different YOLO models for speed vs accuracy."""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.models = {
            'yolov8n': 'yolov8n.pt',      # Nano - fastest, least accurate
            'yolov8s': 'yolov8s.pt',      # Small - good balance
            'yolov8m': 'yolov8m.pt',      # Medium - slower but more accurate
            'yolov5n': 'yolov5nu.pt',     # YOLOv5 nano - alternative
            'yolov5s': 'yolov5su.pt',     # YOLOv5 small
        }
        self.loaded_models = {}
    
    def load_model(self, model_name: str) -> YOLO:
        """Load and cache a YOLO model."""
        if model_name not in self.loaded_models:
            print(f"Loading {model_name}...")
            model = YOLO(self.models[model_name])
            model.to(self.device)
            self.loaded_models[model_name] = model
        return self.loaded_models[model_name]
    
    def benchmark_model(self, model_name: str, test_image: np.ndarray, 
                       iterations: int = 10, imgsz: int = 416) -> Dict:
        """Benchmark a single model."""
        model = self.load_model(model_name)
        
        # Warmup
        for _ in range(3):
            _ = model.predict(test_image, imgsz=imgsz, device=self.device, verbose=False)
        
        # Benchmark
        times = []
        detections_count = []
        
        for i in range(iterations):
            start_time = time.time()
            results = model.predict(test_image, imgsz=imgsz, device=self.device, verbose=False)
            inference_time = time.time() - start_time
            
            times.append(inference_time * 1000)  # Convert to ms
            
            # Count detections
            if len(results) > 0 and results[0].boxes is not None:
                detections_count.append(len(results[0].boxes))
            else:
                detections_count.append(0)
        
        return {
            'model': model_name,
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'avg_detections': np.mean(detections_count),
            'fps': 1000 / np.mean(times)
        }
    
    def compare_models(self, test_image: np.ndarray, imgsz: int = 416) -> List[Dict]:
        """Compare all available models."""
        results = []
        
        print(f"Benchmarking models on {self.device} with image size {imgsz}...")
        
        for model_name in self.models.keys():
            try:
                result = self.benchmark_model(model_name, test_image, imgsz=imgsz)
                results.append(result)
                print(f"{model_name}: {result['avg_time_ms']:.1f}ms avg, {result['fps']:.1f} FPS")
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
        
        # Sort by speed (fastest first)
        results.sort(key=lambda x: x['avg_time_ms'])
        return results
    
    def print_comparison(self, results: List[Dict]):
        """Print formatted comparison table."""
        print("\\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Model':<12} {'Avg Time':<10} {'FPS':<8} {'Min Time':<10} {'Max Time':<10} {'Detections':<11}")
        print("-"*80)
        
        for result in results:
            print(f"{result['model']:<12} "
                  f"{result['avg_time_ms']:<10.1f} "
                  f"{result['fps']:<8.1f} "
                  f"{result['min_time_ms']:<10.1f} "
                  f"{result['max_time_ms']:<10.1f} "
                  f"{result['avg_detections']:<11.1f}")
        
        print("="*80)
        fastest = results[0]
        print(f"✓ FASTEST: {fastest['model']} at {fastest['fps']:.1f} FPS")
        print(f"✓ RECOMMENDATION: Use {fastest['model']} for real-time processing")


class OptimizedObjectGate:
    """Optimized object detection gate with multiple speed improvements."""
    
    def __init__(self, model_name: str = "yolov8n", device: str = "mps", 
                 imgsz: int = 320, conf: float = 0.4, iou: float = 0.7):
        """
        Initialize optimized object gate.
        
        Args:
            model_name: YOLO model to use (yolov8n, yolov8s, etc.)
            device: Device for inference
            imgsz: Input image size (smaller = faster)
            conf: Confidence threshold
            iou: IoU threshold
        """
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        
        print(f"Loading optimized {model_name} model...")
        
        # Model mapping
        models = {
            'yolov8n': 'yolov8n.pt',
            'yolov8s': 'yolov8s.pt', 
            'yolov5n': 'yolov5nu.pt',
            'yolov5s': 'yolov5su.pt',
        }
        
        self.model = YOLO(models.get(model_name, 'yolov8n.pt'))
        
        # Warmup the model
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.model.predict(dummy_img, imgsz=self.imgsz, device=self.device, verbose=False)
        
        print(f"✓ {model_name} model ready on {device}")
    
    def detect_fast(self, frame: np.ndarray) -> Tuple[List, float]:
        """
        Fast object detection with timing.
        
        Returns:
            tuple: (detections, inference_time_ms)
        """
        start_time = time.time()
        
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            stream=False
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'box': box.tolist(),
                    'conf': conf,
                    'cls': cls
                })
        
        return detections, inference_time


def benchmark_current_setup():
    """Benchmark the current object detection setup."""
    import cv2
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some objects to detect
    cv2.rectangle(test_img, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(test_img, (400, 300), 50, (0, 255, 0), -1)
    cv2.putText(test_img, "PERSON", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Benchmark
    benchmark = ModelBenchmark()
    results = benchmark.compare_models(test_img, imgsz=416)
    benchmark.print_comparison(results)
    
    return results


if __name__ == "__main__":
    benchmark_current_setup()