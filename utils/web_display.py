"""
Web-based video display for macOS OpenCV compatibility.
Serves video stream over HTTP for browser viewing.
"""

import cv2
import threading
import time
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True

class VideoStreamHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, video_display=None, **kwargs):
        self.video_display = video_display
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_html_page()
        elif self.path == '/video_feed':
            self.send_video_stream()
        elif self.path == '/status':
            self.send_status()
        elif self.path == '/notifications':
            self.send_notifications()
        elif self.path == '/summary':
            self.send_summary()
        else:
            self.send_error(404)

    def send_html_page(self):
        # Get display info from server
        mode = "summary"
        prompt = ""
        if hasattr(self.server, 'video_display') and self.server.video_display:
            mode = self.server.video_display.mode
            prompt = self.server.video_display.user_prompt
        
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Soteira Video Stream</title>
            <style>
                body {{ 
                    background: #000; 
                    color: #fff; 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                    text-align: center; 
                    margin: 0; 
                    padding: 20px;
                }}
                
                #video {{ 
                    max-width: 90%; 
                    max-height: 60vh; 
                    border: 2px solid #444; 
                    border-radius: 8px;
                }}
                
                #status {{ 
                    margin: 20px; 
                    font-size: 18px; 
                    color: #ccc;
                }}
                
                #prompt-section {{
                    margin: 20px auto;
                    max-width: 800px;
                    padding: 15px;
                    background: #1a1a1a;
                    border-radius: 8px;
                    border-left: 4px solid #0095f6;
                }}
                
                #prompt-label {{
                    font-size: 14px;
                    color: #888;
                    margin-bottom: 8px;
                }}
                
                #prompt-text {{
                    font-size: 16px;
                    color: #fff;
                    font-style: italic;
                }}
                
                .controls {{ 
                    margin: 20px; 
                }}
                
                button {{ 
                    padding: 10px 20px; 
                    font-size: 16px; 
                    margin: 5px; 
                    background: #0095f6;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                }}
                
                button:hover {{
                    background: #007acc;
                }}
                
                /* Instagram-style notifications */
                .notification {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
                    color: white;
                    padding: 16px 20px;
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    font-size: 14px;
                    font-weight: 500;
                    max-width: 350px;
                    animation: slideIn 0.3s ease-out;
                    z-index: 1000;
                    margin-bottom: 10px;
                }}
                
                .notification.alert {{
                    background: linear-gradient(45deg, #ff6b6b, #ff5252);
                }}
                
                .notification-title {{
                    font-weight: bold;
                    margin-bottom: 4px;
                }}
                
                .notification-body {{
                    opacity: 0.9;
                    line-height: 1.4;
                }}
                
                @keyframes slideIn {{
                    from {{ transform: translateX(100%); opacity: 0; }}
                    to {{ transform: translateX(0); opacity: 1; }}
                }}
                
                @keyframes slideOut {{
                    from {{ transform: translateX(0); opacity: 1; }}
                    to {{ transform: translateX(100%); opacity: 0; }}
                }}
                
                /* Summary overlay */
                #summary-overlay {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.8);
                    display: none;
                    align-items: center;
                    justify-content: center;
                    z-index: 2000;
                }}
                
                #summary-content {{
                    background: #1a1a1a;
                    padding: 30px;
                    border-radius: 12px;
                    max-width: 80%;
                    max-height: 80%;
                    overflow-y: auto;
                    border: 1px solid #333;
                }}
                
                #summary-title {{
                    font-size: 24px;
                    margin-bottom: 20px;
                    color: #0095f6;
                }}
                
                #summary-text {{
                    font-size: 16px;
                    line-height: 1.6;
                    text-align: left;
                    white-space: pre-wrap;
                }}
                
                .close-btn {{
                    float: right;
                    background: #ff4444;
                    border: none;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Soteira Real-time Video Analysis</h1>
            <div id="status">Loading...</div>
            <img id="video" src="/video_feed" alt="Video Stream">
            
            <div id="prompt-section">
                <div id="prompt-label">Current Analysis Prompt:</div>
                <div id="prompt-text">"{prompt}"</div>
            </div>
            
            <div class="controls">
                <button onclick="window.location.reload()">Refresh</button>
                <button onclick="toggleFullscreen()">Fullscreen</button>
                {"<button onclick='showSummary()'>Show Summary</button>" if mode == "summary" else ""}
            </div>
            
            <!-- Summary overlay -->
            <div id="summary-overlay">
                <div id="summary-content">
                    <button class="close-btn" onclick="hideSummary()">Close</button>
                    <div id="summary-title">Video Analysis Summary</div>
                    <div id="summary-text">Loading summary...</div>
                </div>
            </div>
            
            <script>
                let notificationId = 0;
                let currentMode = "{mode}";
                
                function toggleFullscreen() {{
                    const video = document.getElementById('video');
                    if (video.requestFullscreen) video.requestFullscreen();
                }}
                
                function showNotification(title, body, isAlert = false) {{
                    const notification = document.createElement('div');
                    notification.className = `notification ${{isAlert ? 'alert' : ''}}`;
                    notification.id = `notification-${{notificationId++}}`;
                    
                    notification.innerHTML = `
                        <div class="notification-title">${{title}}</div>
                        <div class="notification-body">${{body}}</div>
                    `;
                    
                    document.body.appendChild(notification);
                    
                    // Auto-remove after 5 seconds
                    setTimeout(() => {{
                        if (notification.parentNode) {{
                            notification.style.animation = 'slideOut 0.3s ease-out';
                            setTimeout(() => {{
                                if (notification.parentNode) {{
                                    notification.parentNode.removeChild(notification);
                                }}
                            }}, 300);
                        }}
                    }}, 5000);
                }}
                
                function showSummary() {{
                    fetch('/summary')
                        .then(r => r.json())
                        .then(data => {{
                            document.getElementById('summary-text').innerText = data.summary || 'No summary available yet.';
                            document.getElementById('summary-overlay').style.display = 'flex';
                        }})
                        .catch(() => {{
                            document.getElementById('summary-text').innerText = 'Error loading summary.';
                            document.getElementById('summary-overlay').style.display = 'flex';
                        }});
                }}
                
                function hideSummary() {{
                    document.getElementById('summary-overlay').style.display = 'none';
                }}
                
                // Status updates
                setInterval(() => {{
                    fetch('/status')
                        .then(r => r.json())
                        .then(data => {{
                            document.getElementById('status').innerHTML = 
                                `FPS: ${{data.fps}} | Frames: ${{data.frames}} | Status: ${{data.status}} | Mode: ${{currentMode.toUpperCase()}}`;
                            
                            // Check for video end in summary mode
                            if (currentMode === 'summary' && data.video_ended) {{
                                showSummary();
                            }}
                        }})
                        .catch(() => {{}});
                }}, 1000);
                
                // Check for new notifications in alert mode
                if (currentMode === 'alert') {{
                    setInterval(() => {{
                        fetch('/notifications')
                            .then(r => r.json())
                            .then(data => {{
                                data.new_notifications.forEach(notif => {{
                                    showNotification('🚨 Alert Detection', notif.message, true);
                                }});
                            }})
                            .catch(() => {{}});
                    }}, 2000);
                }}
                
                // Close summary with Escape key
                document.addEventListener('keydown', (e) => {{
                    if (e.key === 'Escape') {{
                        hideSummary();
                    }}
                }});
            </script>
        </body>
        </html>
        '''
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def send_video_stream(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        
        last_frame_time = time.time()
        
        while True:
            try:
                if hasattr(self.server, 'video_display') and self.server.video_display:
                    current_time = time.time()
                    
                    # Control display rate based on target FPS
                    time_since_last = current_time - last_frame_time
                    if time_since_last >= self.server.video_display.frame_interval:
                        frame = self.server.video_display.get_latest_frame()
                        if frame is not None:
                            # Encode frame as JPEG
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            
                            # Send frame
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(buffer)))
                            self.end_headers()
                            self.wfile.write(buffer)
                            self.wfile.write(b'\r\n')
                            
                            last_frame_time = current_time
                        else:
                            time.sleep(0.01)  # Short sleep if no frame
                    else:
                        # Sleep for remaining time to maintain target FPS
                        sleep_time = self.server.video_display.frame_interval - time_since_last
                        time.sleep(max(0.001, sleep_time))
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[WEB_DISPLAY] Stream error: {e}")
                break

    def send_status(self):
        if hasattr(self.server, 'video_display') and self.server.video_display:
            status = self.server.video_display.get_status()
        else:
            status = {"fps": 0, "frames": 0, "status": "disconnected", "video_ended": False}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def send_notifications(self):
        if hasattr(self.server, 'video_display') and self.server.video_display:
            notifications = self.server.video_display.get_new_notifications()
        else:
            notifications = []
        
        response = {"new_notifications": notifications}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def send_summary(self):
        if hasattr(self.server, 'video_display') and self.server.video_display:
            summary = self.server.video_display.get_summary()
        else:
            summary = "No summary available."
        
        response = {"summary": summary}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass

class WebVideoDisplay:
    def __init__(self, port=8888, target_fps=30.0, user_prompt="", mode="summary"):
        self.port = port
        self.target_fps = target_fps
        self.user_prompt = user_prompt
        self.mode = mode
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.fps_counter = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.running = False
        self.server = None
        self.server_thread = None
        
        # Frame timing for consistent display rate
        self.frame_interval = 1.0 / target_fps
        self.last_display_time = 0
        
        # Notifications and results
        self.notifications = []  # List of alert notifications
        self.notifications_lock = threading.Lock()
        self.final_summary = ""
        self.video_ended = False

    def start(self):
        """Start the web server for video display."""
        try:
            # Create server with custom handler
            def handler(*args, **kwargs):
                return VideoStreamHandler(*args, video_display=self, **kwargs)
            
            self.server = ThreadedHTTPServer(('localhost', self.port), handler)
            self.server.video_display = self
            
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            self.running = True
            
            print(f"[WEB_DISPLAY] Started web server at http://localhost:{self.port}")
            print(f"[WEB_DISPLAY] Open this URL in your browser to view the video stream")
            return True
            
        except Exception as e:
            print(f"[WEB_DISPLAY] Failed to start web server: {e}")
            return False

    def stop(self):
        """Stop the web server."""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=2)
        print("[WEB_DISPLAY] Web server stopped")

    def update_frame(self, frame):
        """Update the latest frame for streaming."""
        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.frame_count += 1
            
            # Update FPS counter
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps_counter = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now

    def get_latest_frame(self):
        """Get the latest frame for streaming."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_status(self):
        """Get current status for web interface."""
        return {
            "fps": round(self.fps_counter, 1),
            "frames": self.frame_count,
            "status": "connected" if self.running else "disconnected",
            "video_ended": self.video_ended
        }

    def add_notification(self, message):
        """Add a new alert notification."""
        with self.notifications_lock:
            self.notifications.append({
                "message": message,
                "timestamp": time.time(),
                "read": False
            })

    def get_new_notifications(self):
        """Get unread notifications and mark them as read."""
        with self.notifications_lock:
            new_notifications = [notif for notif in self.notifications if not notif["read"]]
            # Mark as read
            for notif in self.notifications:
                notif["read"] = True
            return new_notifications

    def set_summary(self, summary):
        """Set the final summary."""
        self.final_summary = summary
        print(f"[WEB_DISPLAY] Summary set: {len(summary)} characters")

    def get_summary(self):
        """Get the final summary."""
        print(f"[WEB_DISPLAY] Summary requested: {len(self.final_summary)} characters available")
        return self.final_summary

    def set_video_ended(self, ended=True):
        """Mark video as ended."""
        self.video_ended = ended