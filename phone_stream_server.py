#!/usr/bin/env python3
"""
Phone stream server that serves frames via a simple HTTP endpoint.
This bridges the WebRTC frames to a format that main.py can consume.
"""

import asyncio
import json
import threading
import time
import queue
from aiohttp import web
import cv2
import numpy as np


class PhoneStreamServer:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_timestamp = 0
        self.frame_count = 0
        
    def receive_frame(self, img):
        """Receive a frame directly from the unified frame handler."""
        with self.frame_lock:
            self.latest_frame = img.copy()
            self.frame_timestamp = time.time()
            self.frame_count += 1
        
        # Log every 30th frame to avoid spam
        if self.frame_count % 30 == 0:
            print(f"[STREAM_SERVER] Received frame #{self.frame_count}, shape: {img.shape}")
    
    
    async def get_frame(self, request):
        """HTTP endpoint to get the latest frame as JPEG."""
        with self.frame_lock:
            if self.latest_frame is None:
                return web.Response(status=404, text="No frame available")
            
            # Check if frame is fresh (increased to 5 seconds to handle temporary pauses)
            frame_age = time.time() - self.frame_timestamp
            if frame_age > 5.0:
                return web.Response(status=408, text="Frame too old")
            
            # Encode frame as JPEG with optimized settings for web streaming
            success, buffer = cv2.imencode('.jpg', self.latest_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 85,  # Slightly lower quality for better speed
                cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Enable JPEG optimization
            ])
            if not success:
                return web.Response(status=500, text="Failed to encode frame")
            
            # Log less frequently to reduce overhead
            if self.frame_count % 30 == 0:
                print(f"[STREAM_SERVER] HTTP request: Serving frame {self.latest_frame.shape}, age {frame_age:.3f}s")
            
        return web.Response(
            body=buffer.tobytes(), 
            content_type='image/jpeg',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    
    async def get_stats(self, request):
        """HTTP endpoint to get capture statistics."""
        with self.frame_lock:
            stats = {
                'frame_count': self.frame_count,
                'latest_frame_age': time.time() - self.frame_timestamp if self.latest_frame is not None else -1,
                'has_frame': self.latest_frame is not None
            }
        
        return web.json_response(stats)
    
    def get_app(self):
        """Create and return the aiohttp application."""
        app = web.Application()
        app.router.add_get('/frame', self.get_frame)
        app.router.add_get('/stats', self.get_stats)
        return app


# Global instance
_stream_server = PhoneStreamServer()

def get_stream_server():
    return _stream_server