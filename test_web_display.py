#!/usr/bin/env python3
"""
Test script for web display functionality.
Creates a simple video stream with generated frames.
"""

import time
import numpy as np
import threading
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    from utils.web_display import WebVideoDisplay
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install dependencies first:")
    print("pip3 install opencv-python numpy")
    sys.exit(1)

def generate_test_frame(frame_num):
    """Generate a test frame with moving elements."""
    # Create 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add moving circle
    center_x = int(320 + 200 * np.sin(frame_num * 0.05))
    center_y = int(240 + 100 * np.cos(frame_num * 0.03))
    cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
    
    # Add frame counter text
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add timestamp
    cv2.putText(frame, f"Time: {time.time():.1f}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Add test pattern
    for i in range(0, 640, 40):
        cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
    for i in range(0, 480, 40):
        cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)
    
    return frame

def main():
    print("Testing Web Display System")
    print("=" * 40)
    
    # Create web display
    web_display = WebVideoDisplay(port=8888)
    
    # Start web server
    if not web_display.start():
        print("Failed to start web display server")
        return
    
    print("\n✅ Web display server started successfully!")
    print(f"🌐 Open your browser to: http://localhost:8888")
    print("📺 You should see a moving green circle with frame counter")
    print("🛑 Press Ctrl+C to stop")
    
    frame_num = 0
    fps = 30
    frame_time = 1.0 / fps
    
    try:
        while True:
            start_time = time.time()
            
            # Generate test frame
            frame = generate_test_frame(frame_num)
            
            # Send to web display
            web_display.update_frame(frame)
            
            frame_num += 1
            
            # Print status every 60 frames
            if frame_num % 60 == 0:
                status = web_display.get_status()
                print(f"📊 Status - Frame: {frame_num}, "
                      f"Web FPS: {status['fps']}, "
                      f"Status: {status['status']}")
            
            # Sleep to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping test...")
    
    # Cleanup
    web_display.stop()
    print("✅ Test completed")

if __name__ == "__main__":
    main()