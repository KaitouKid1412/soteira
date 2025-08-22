#!/usr/bin/env python3
"""
HTTP-based phone capture that fetches frames from the video server.
"""

import requests
import cv2
import numpy as np
import time
from typing import Optional, Tuple


class PhoneHTTPCapture:
    """
    A cv2.VideoCapture-like interface that fetches frames from the video server via HTTP.
    """
    
    def __init__(self, server_url: str = "https://localhost:8443"):
        self.server_url = server_url.rstrip('/')
        self.frame_url = f"{self.server_url}/stream/frame"
        self.stats_url = f"{self.server_url}/stream/stats"
        self.session = requests.Session()
        self.session.verify = False  # Ignore SSL certificate warnings for self-signed cert
        
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.is_opened = False
        self.frame_count = 0
        self.last_successful_fetch = 0
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if we can connect to the server."""
        try:
            response = self.session.get(self.stats_url, timeout=5)
            if response.status_code == 200:
                self.is_opened = True
                print("[PHONE_HTTP] Successfully connected to video server")
            else:
                print(f"[PHONE_HTTP] Server responded with status {response.status_code}")
        except Exception as e:
            print(f"[PHONE_HTTP] Failed to connect to server: {e}")
            self.is_opened = False
    
    def isOpened(self) -> bool:
        """Check if the phone capture is available."""
        return self.is_opened
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the phone stream via HTTP.
        Returns (success, frame) tuple like cv2.VideoCapture.read()
        """
        if not self.is_opened:
            return False, None
        
        try:
            # Fetch frame from server with reduced timeout for better responsiveness
            response = self.session.get(self.frame_url, timeout=0.5)
            
            if response.status_code == 200:
                # Decode JPEG data
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.frame_count += 1
                    self.last_successful_fetch = time.time()
                    return True, frame
                else:
                    print("[PHONE_HTTP] Failed to decode frame")
                    return False, None
            
            elif response.status_code == 404:
                # No frame available yet - don't log to reduce noise
                return False, None
            
            elif response.status_code == 408:
                # Frame too old - don't log to reduce noise
                return False, None
            
            else:
                print(f"[PHONE_HTTP] Unexpected status code: {response.status_code} - {response.text}")
                return False, None
                
        except requests.exceptions.Timeout:
            # Timeout is expected when no new frames - don't log to reduce noise
            return False, None
        except Exception as e:
            print(f"[PHONE_HTTP] Error fetching frame: {e}")
            return False, None
    
    def get(self, prop_id):
        """
        Get video capture properties (cv2.VideoCapture compatible).
        For phone streams, we return sensible defaults.
        """
        import cv2
        
        if prop_id == cv2.CAP_PROP_FPS:
            # Phone streams typically run at 30 FPS
            return 30.0
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            # Default phone resolution width
            return 1920
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            # Default phone resolution height  
            return 1080
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            # Phone streams are infinite, return -1
            return -1
        else:
            # Return 0 for unsupported properties
            return 0
    
    def release(self):
        """Release the phone capture resources."""
        self.session.close()
        self.is_opened = False
        print(f"[PHONE_HTTP] Released. {self.frame_count} frames read")
    
    def get_stats(self) -> dict:
        """Get statistics from the server."""
        try:
            response = self.session.get(self.stats_url, timeout=2)
            if response.status_code == 200:
                stats = response.json()
                stats['frames_read_http'] = self.frame_count
                stats['last_successful_fetch'] = self.last_successful_fetch
                return stats
            else:
                return {'error': f'Status code {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}