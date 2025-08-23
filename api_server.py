#!/usr/bin/env python3
"""
FastAPI backend server for Soteira video analysis system.
Provides REST API and WebSocket endpoints for real-time video processing.
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from numpy import False_
from pydantic import BaseModel
import uvicorn

from main import VideoGatingSystem
from utils.video_streamer import VideoStreamer
import argparse


class ProcessingConfig(BaseModel):
    video_path: str
    mode: str = "summary"  # "summary", "alert", or "analysis"
    prompt: str = "Analyze this video for security threats"
    motion_thresh: float = 0.15
    conf: float = 0.4
    imgsz: int = 320
    scene_hist: float = 0.7
    scene_ssim: float = 0.85
    buffer_frames: int = 3
    similarity_threshold: float = 0.85
    disable_quality_filter: bool = True
    skip_scene: bool = False
    stop_on_good_frame: bool = True
    streaming_mode: bool = False  # Enable Gemini Flash 2.5 streaming


class ProcessingStatus(BaseModel):
    is_running: bool
    fps: float
    frames_processed: int
    total_frames: int
    mode: str
    current_prompt: str
    alerts_count: int
    detections_count: int


class AlertNotification(BaseModel):
    id: str
    timestamp: datetime
    message: str
    confidence: float
    frame_path: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str


class VideoAnalysisServer:
    def __init__(self):
        self.app = FastAPI(title="Soteira Video Analysis API", version="1.0.0")
        self.setup_routes()
        
        # Processing state
        self.video_system: Optional[VideoGatingSystem] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self.current_config: Optional[ProcessingConfig] = None
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        
        # Results storage
        self.alerts: List[AlertNotification] = []
        self.processing_stats = {
            "fps": 0.0,
            "frames_processed": 0,
            "total_frames": 0,
            "detections_count": 0
        }
        
        # Queue for pending alerts to broadcast
        self.pending_alerts_queue = []
        self.last_alert_broadcast_check = 0
        
        # Queue for streaming tokens
        self.pending_token_queue = []
        
    def get_video_presets(self):
        """Get predefined video configurations."""
        return {
            "baby.mov": ProcessingConfig(
                video_path="videos/baby.mov",
                mode="alert",
                prompt="Alert me if my babies are in danger ?",
                motion_thresh=0.001,
                conf=0.4,
                imgsz=320,
                scene_hist=0.5,  # Default since --skip-scene is True
                scene_ssim=0.5,  # Default since --skip-scene is True  
                buffer_frames=3,
                similarity_threshold=0.9,
                disable_quality_filter=True,
                skip_scene=True
            ),
            "desktop.mov": ProcessingConfig(
                video_path="videos/desktop.mov",
                mode="summary", 
                prompt="What action is the user performing in this video ?",
                motion_thresh=0.001,
                conf=0.3,
                imgsz=320,  # Default since not specified in command
                scene_hist=0.7,  # Default since --skip-scene is True
                scene_ssim=0.85,  # Default since --skip-scene is True
                buffer_frames=3,
                similarity_threshold=0.75,  # Optimized for speed
                disable_quality_filter=True,
                skip_scene=False,
                stop_on_good_frame=True
            ),
            "restaurant.mov": ProcessingConfig(
                video_path="videos/restaurant.mov",
                mode="alert",
                prompt="Alert me whenever there is a food safety standard violation in the video",
                motion_thresh=0.2,
                conf=0.7,
                imgsz=256,
                scene_hist=0.3,
                scene_ssim=0.7,
                buffer_frames=2,
                similarity_threshold=0.65,  # Default since not specified in command
                disable_quality_filter=True,
                skip_scene=False,  # Note: restaurant command doesn't use --skip-scene
                stop_on_good_frame=True
            ),
            "phone_stream": ProcessingConfig(
                video_path="phone",
                mode="realtime_description",
                prompt="Provide clear, concise scene descriptions for accessibility. Focus on: people and their activities, objects and locations, any movement or changes. Keep descriptions brief but informative.",
                motion_thresh=0.02,
                conf=0.4,
                imgsz=416,
                scene_hist=0.50,
                scene_ssim=0.45,
                buffer_frames=5,
                similarity_threshold=0.9,
                disable_quality_filter=False,
                skip_scene=False,
                stop_on_good_frame=True
            ),
            "movie": ProcessingConfig(
                video_path="videos/movie.mp4",
                mode="alert",
                prompt="You are an expert movie narrator. You are being sent a sequence of deduplicated frames from a movie in real time. Your job is to narrate what's happening on screen as if telling a story, adding context from previous frames to make it cohesive and engaging.\nInstructions:\nContext Awareness: Maintain a running understanding of previous frames to narrate the story naturally, even if the current frame has minimal change.\nNo Repetition: Do not restate things already clearly described earlier unless they have meaningfully changed or progressed.\nNarrative Focus: Prioritize describing character actions, emotions, setting changes, and plot developments over static details.\nConciseness: Be vivid but efficient, avoiding redundant or filler text.\nStorytelling Tone: Make it sound like a smooth narration for a movie audience, not a frame-by-frame commentary.\nReal-Time Adaptation: Assume frames may skip minor transitions—fill small gaps logically, maintaining narrative flow.\nNo Guesswork: Avoid inventing details not supported by visual evidence, but infer obvious next actions (e.g., someone raising a hand to knock).\nContinuity Memory: Remember character names, locations, and scene context from earlier frames for seamless narration.\nIgnore Deduplication Artifacts: If a frame looks similar to a previous one, briefly acknowledge continuity instead of describing the same static content again.\nYour output should feel like a natural ongoing movie narration, not repetitive frame labeling.",
                motion_thresh=0.001,
                conf=0.3,
                imgsz=320,
                scene_hist=0.7,
                scene_ssim=0.85,
                buffer_frames=3,
                similarity_threshold=0.9,
                disable_quality_filter=True,
                skip_scene=True,
                stop_on_good_frame=True
            )
        }

    def setup_routes(self):
        """Setup FastAPI routes."""
        
        # Serve static frontend files
        self.app.mount("/static", StaticFiles(directory="frontend"), name="static")
        
        @self.app.get("/")
        async def root():
            """Serve the main frontend page."""
            with open("frontend/index.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        
        @self.app.get("/app.js")
        async def serve_js():
            """Serve the JavaScript file."""
            with open("frontend/app.js", "r") as f:
                js_content = f.read()
            return Response(content=js_content, media_type="application/javascript")
        
        @self.app.get("/api/status")
        async def api_root():
            return {"message": "Soteira Video Analysis API", "status": "ready"}
        
        @self.app.get("/status")
        async def get_status() -> ProcessingStatus:
            """Get current processing status."""
            return ProcessingStatus(
                is_running=self.is_processing,
                fps=self.processing_stats["fps"],
                frames_processed=self.processing_stats["frames_processed"],
                total_frames=self.processing_stats["total_frames"],
                mode=self.current_config.mode if self.current_config else "none",
                current_prompt=self.current_config.prompt if self.current_config else "",
                alerts_count=len(self.alerts),
                detections_count=self.processing_stats["detections_count"]
            )
        
        @self.app.post("/start_processing")
        async def start_processing(config: ProcessingConfig):
            """Start video processing with given configuration."""
            if self.is_processing:
                raise HTTPException(status_code=400, detail="Processing already running")
            
            # Check if this matches a known preset and use the full preset config
            presets = self.get_video_presets()
            for preset_name, preset_config in presets.items():
                if preset_config.video_path == config.video_path:
                    print(f"[API] Detected preset usage: {preset_name} for {config.video_path}")
                    # Use preset config but override with any explicit values from request
                    preset_dict = preset_config.model_dump()
                    preset_dict['mode'] = config.mode  # Override with user's mode selection
                    preset_dict['prompt'] = config.prompt  # Override with user's prompt
                    preset_dict['streaming_mode'] = config.streaming_mode  # Preserve streaming mode setting
                    config = ProcessingConfig(**preset_dict)
                    print(f"[API] Applied preset config: skip_scene={config.skip_scene}, streaming_mode={config.streaming_mode}")
                    break
            
            # Validate video file exists (skip for phone stream)
            if config.video_path != "phone" and not Path(config.video_path).exists():
                raise HTTPException(status_code=404, detail="Video file not found")
            
            # Reset previous processing if any
            if self.is_processing and self.video_system:
                print("[API] Stopping previous processing...")
                self.is_processing = False
                self.video_system.running = False
                if hasattr(self.video_system, 'video_streamer') and self.video_system.video_streamer:
                    self.video_system.video_streamer.stop_streaming()
                
                # Wait a moment for cleanup
                import time
                time.sleep(1)
            
            self.current_config = config
            self.alerts.clear()
            self.processing_stats = {"fps": 0.0, "frames_processed": 0, "total_frames": 0, "detections_count": 0}
            self.video_system = None  # Reset video system
            
            # Start processing in background thread
            self.processing_thread = threading.Thread(target=self._run_processing, daemon=True)
            self.processing_thread.start()
            
            return {"message": "Processing started", "config": config}
        
        @self.app.post("/stop_processing")
        async def stop_processing():
            """Stop current video processing."""
            if not self.is_processing:
                raise HTTPException(status_code=400, detail="No processing running")
            
            print("[API] Stop processing requested")
            self.is_processing = False
            if self.video_system:
                self.video_system.running = False
                print(f"[API] Set video_system.running = False")
                
                # Also stop the video streamer if it exists
                if hasattr(self.video_system, 'video_streamer') and self.video_system.video_streamer:
                    self.video_system.video_streamer.stop_streaming()
                    print(f"[API] Stopped video streamer")
            
            return {"message": "Processing stopped"}
        
        @self.app.get("/alerts")
        async def get_alerts() -> List[AlertNotification]:
            """Get all alerts from current session."""
            print(f"[API] GET /alerts called, returning {len(self.alerts)} alerts")
            return self.alerts
        
        @self.app.get("/video_presets")
        async def get_video_presets():
            """Get available video presets."""
            presets = self.get_video_presets()
            return {name: config.model_dump() for name, config in presets.items()}
        
        @self.app.get("/summary")
        async def get_summary():
            """Get video analysis summary."""
            if self.video_system and hasattr(self.video_system, 'llm_sink'):
                summary = self.video_system.llm_sink.generate_synthesis_summary()
                return {"summary": summary}
            return {"summary": "No summary available."}
        
        @self.app.post("/ask_question")
        async def ask_question(request: QuestionRequest):
            """Ask a question about the processed video."""
            if not self.video_system or not hasattr(self.video_system, 'llm_sink'):
                raise HTTPException(status_code=404, detail="No video processing system available")
            
            if not self.current_config:
                raise HTTPException(status_code=400, detail="No video has been processed yet")
            
            if self.current_config.mode not in ["alert", "summary"]:
                raise HTTPException(status_code=400, detail="Q&A only available in alert and summary modes")
            
            try:
                answer = self.video_system.llm_sink.answer_question(request.question)
                return {"question": request.question, "answer": answer}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
        
        @self.app.get("/video_feed")
        async def video_feed():
            """Stream current video frames."""
            # Check for phone stream first - it may be available even when processing is paused
            if (self.current_config and self.current_config.video_path == "phone"):
                return await self.proxy_phone_video_feed()
            
            if not self.is_processing:
                raise HTTPException(status_code=404, detail="No active video processing")
            
            # Phone stream already handled above
            
            # For non-phone sources, use the web display
            if not self.video_system or not hasattr(self.video_system, 'web_display') or not self.video_system.web_display:
                raise HTTPException(status_code=404, detail="No web display available")
            
            def generate_frames():
                try:
                    while self.is_processing and self.video_system and self.video_system.web_display:
                        frame = self.video_system.web_display.get_latest_frame()
                        if frame is not None:
                            import cv2
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        else:
                            time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"[API] Video feed error: {e}")
            
            return StreamingResponse(generate_frames(), 
                                   media_type="multipart/x-mixed-replace; boundary=frame")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.connect_websocket(websocket)
            try:
                while True:
                    # Check for pending streaming tokens
                    if self.pending_token_queue:
                        print(f"[WS] Found {len(self.pending_token_queue)} tokens in queue")
                        while self.pending_token_queue:
                            token_data = self.pending_token_queue.pop(0)
                            print(f"[WS] 🚀 Broadcasting token: '{token_data['token']}'")
                            await websocket.send_json({
                                "type": "token_stream",
                                "data": {
                                    "token": token_data['token'],
                                    "timestamp": token_data['timestamp']
                                }
                            })
                            print(f"[WS] ✅ Token broadcast complete")
                    else:
                        # Periodic check to show queue is empty (only occasionally)
                        import time
                        if not hasattr(self, '_last_empty_log') or time.time() - self._last_empty_log > 10:
                            print(f"[WS] Token queue empty")
                            self._last_empty_log = time.time()
                    
                    # Check for pending alerts to broadcast
                    if self.pending_alerts_queue:
                        while self.pending_alerts_queue:
                            alert = self.pending_alerts_queue.pop(0)
                            print(f"[WS] Broadcasting queued alert: {alert.message}")
                            # Convert datetime to ISO string for JSON serialization
                            alert_data = {
                                "id": alert.id,
                                "timestamp": alert.timestamp.isoformat(),
                                "message": alert.message,
                                "confidence": alert.confidence,
                                "frame_path": alert.frame_path
                            }
                            await websocket.send_json({
                                "type": "new_alert",
                                "data": alert_data
                            })
                            print(f"[WS] Alert broadcast complete")
                    
                    # Send periodic status updates (less frequent for performance)
                    if not hasattr(self, '_last_status_time') or time.time() - self._last_status_time > 5:
                        status = await get_status()
                        await websocket.send_json({
                            "type": "status_update",
                            "data": status.model_dump()
                        })
                        self._last_status_time = time.time()
                    
                    # Very short sleep for real-time token streaming
                    await asyncio.sleep(0.1)  # 100ms for near real-time
                    
            except WebSocketDisconnect:
                self.disconnect_websocket(websocket)
    
    async def proxy_phone_video_feed(self):
        """Proxy video feed from the video server for phone streams."""
        try:
            import aiohttp
        except ImportError:
            raise HTTPException(status_code=500, detail="aiohttp not available for phone video streaming")
        
        async def generate_phone_frames():
            retry_count = 0
            max_retries = 5
            
            try:
                # Connect to the video server's stream endpoint
                timeout = aiohttp.ClientTimeout(total=None, sock_read=2, sock_connect=5)
                connector = aiohttp.TCPConnector(verify_ssl=False, limit=10, limit_per_host=10)
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    while self.current_config and self.current_config.video_path == "phone":
                        try:
                            async with session.get("https://localhost:8443/stream/frame") as response:
                                if response.status == 200:
                                    frame_data = await response.read()
                                    retry_count = 0  # Reset retry count on success
                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                                elif response.status == 408:  # Timeout - frame too old
                                    await asyncio.sleep(0.05)
                                elif response.status == 404:  # No frame available
                                    await asyncio.sleep(0.033)
                                else:
                                    print(f"[API] Unexpected response status: {response.status}")
                                    await asyncio.sleep(0.1)
                        except asyncio.TimeoutError:
                            retry_count += 1
                            if retry_count > max_retries:
                                print(f"[API] Phone proxy timeout after {max_retries} retries")
                                break
                            await asyncio.sleep(0.5)
                        except Exception as e:
                            retry_count += 1
                            print(f"[API] Phone proxy error: {e}")
                            if retry_count > max_retries:
                                print(f"[API] Phone proxy failed after {max_retries} retries")
                                break
                            await asyncio.sleep(0.2)
            except Exception as e:
                print(f"[API] Phone proxy session error: {e}")
        
        return StreamingResponse(generate_phone_frames(), 
                               media_type="multipart/x-mixed-replace; boundary=frame")
    
    async def connect_websocket(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast_alert(self, alert: AlertNotification):
        """Broadcast alert to all connected WebSocket clients."""
        message = {
            "type": "new_alert",
            "data": alert.model_dump()
        }
        
        print(f"[WS] Broadcasting alert to {len(self.active_connections)} connections")
        print(f"[WS] Alert message: {alert.message}")
        
        # Remove disconnected clients
        connected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                connected.append(connection)
                print(f"[WS] Successfully sent alert to WebSocket client")
            except Exception as e:
                print(f"[WS] Failed to send to WebSocket client: {e}")
        
        self.active_connections = connected
        print(f"[WS] Active connections after broadcast: {len(self.active_connections)}")
    
    def _run_processing(self):
        """Run video processing in background thread."""
        self.is_processing = True
        
        try:
            # Create required directories
            Path("./processing_output").mkdir(exist_ok=True)
            Path("./events").mkdir(exist_ok=True)
            Path("./events/llm_analysis").mkdir(exist_ok=True, parents=True)
            
            # Log all configuration parameters being used
            print(f"{'='*80}")
            print(f"VIDEO PROCESSING CONFIGURATION")
            print(f"{'='*80}")
            print(f"Video Path: {self.current_config.video_path}")
            print(f"Mode: {self.current_config.mode}")
            print(f"Prompt: {self.current_config.prompt}")
            print(f"Motion Threshold: {self.current_config.motion_thresh}")
            print(f"Scene Histogram: {self.current_config.scene_hist}")
            print(f"Scene SSIM: {self.current_config.scene_ssim}")
            print(f"Skip Scene: {self.current_config.skip_scene}")
            print(f"Confidence: {self.current_config.conf}")
            print(f"Image Size: {self.current_config.imgsz}")
            print(f"Buffer Frames: {self.current_config.buffer_frames}")
            print(f"Similarity Threshold: {self.current_config.similarity_threshold}")
            print(f"Disable Quality Filter: {self.current_config.disable_quality_filter}")
            print(f"Stop on Good Frame: {self.current_config.stop_on_good_frame}")
            print(f"Stream Loop: False (hardcoded)")
            print(f"No Stream Loop: True (hardcoded)")
            print(f"{'='*80}")

            # Create arguments object similar to main.py
            args = argparse.Namespace(
                # Basic config
                prompt=self.current_config.prompt,
                save_dir="./processing_output",
                source=self.current_config.video_path,
                
                # Video streaming
                stream_video=True,
                stream_speed=1.0,
                stream_loop=False,
                no_stream_loop=True,
                
                # OpenAI config
                openai_api_key="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",
                openai_model="gpt-4o",
                dry_run=False,  # Enable real LLM calls
                
                # Gate thresholds
                motion_thresh=self.current_config.motion_thresh,
                scene_hist=self.current_config.scene_hist,
                scene_ssim=self.current_config.scene_ssim,
                skip_scene=self.current_config.skip_scene,
                
                # Object detection
                confirm_hits=2,
                max_age_seconds=60.0,
                imgsz=self.current_config.imgsz,
                conf=self.current_config.conf,
                iou=0.5,
                min_area_frac=0.01,
                classes=None,
                
                # Quality filtering
                blur_threshold=25.0,
                min_brightness=20.0,
                max_brightness=240.0,
                min_contrast=10.0,
                buffer_frames=self.current_config.buffer_frames,
                disable_quality_filter=self.current_config.disable_quality_filter,
                stop_on_good_frame=self.current_config.stop_on_good_frame,
                capture_all_frames=False,
                
                # Display and web
                web_port=8888,
                display_fps=None,
                no_display=False,
                
                # LLM processing
                llm_workers=8,  # Increased for better real-time performance
                similarity_threshold=self.current_config.similarity_threshold,
                mode=self.current_config.mode,
                streaming_mode=self.current_config.streaming_mode,
                gemini_api_key=os.getenv('GEMINI_API_KEY'),
                
                # Performance
                enable_60fps_mode=False,
                target_fps=30.0,
                min_fps=10.0,
                max_fps=60.0,
                bypass_gates=False,
                
                # Debugging
                debug=False
            )
            
            # Create and run video system
            print(f"[API] Creating VideoGatingSystem with video: {self.current_config.video_path}")
            self.video_system = VideoGatingSystem(args)
            print(f"[API] VideoGatingSystem created, web_display: {hasattr(self.video_system, 'web_display')}")
            
            # Setup callback for alerts (if in alert or realtime_description mode)
            if self.current_config.mode in ["alert", "realtime_description"]:
                def alert_callback(message):
                    alert = AlertNotification(
                        id=str(len(self.alerts)),
                        timestamp=datetime.now(),
                        message=message,
                        confidence=0.8  # Default confidence
                    )
                    self.alerts.append(alert)
                    print(f"[API] New alert added: {message}")
                    
                    # Store alert for HTTP endpoint
                    # Note: WebSocket broadcasting would need proper async handling
                
                # Create a web display wrapper that captures alerts
                class AlertCapture:
                    def __init__(self, server):
                        self.server = server
                        print(f"[API] AlertCapture created")
                    
                    def add_notification(self, message):
                        print(f"[API] AlertCapture.add_notification called with: {message}")
                        alert = AlertNotification(
                            id=str(len(self.server.alerts)),
                            timestamp=datetime.now(),
                            message=message,
                            confidence=0.8  # Default confidence
                        )
                        self.server.alerts.append(alert)
                        print(f"[API] Alert added to server.alerts, total count: {len(self.server.alerts)}")
                        
                        # Add to queue for WebSocket broadcasting
                        self.server.pending_alerts_queue.append(alert)
                        print(f"[API] Alert added to broadcast queue. Queue size: {len(self.server.pending_alerts_queue)}")
                
                # Set the alert capture on LLM sink
                if hasattr(self.video_system, 'llm_sink'):
                    alert_capture = AlertCapture(self)
                    self.video_system.llm_sink.web_display = alert_capture
                    print(f"[API] Alert capture set for LLM sink")
                    print(f"[API] LLM sink web_display is now: {self.video_system.llm_sink.web_display}")
                else:
                    print(f"[API] ERROR: video_system has no llm_sink attribute!")
            
            # CRITICAL FIX: Replace web_display with AlertCapture wrapper AFTER VideoGatingSystem init
            print(f"[API DEBUG] Mode: {self.current_config.mode}, has llm_sink: {hasattr(self.video_system, 'llm_sink')}")
            if self.current_config.mode in ["alert", "realtime_description"] and hasattr(self.video_system, 'llm_sink'):
                print(f"[API DEBUG] Entering alert wrapper setup...")
                class AlertCaptureWrapper:
                    def __init__(self, server, original_web_display):
                        self.server = server
                        self.original_web_display = original_web_display
                        print(f"[API] AlertCaptureWrapper created")
                    
                    def add_notification(self, message):
                        print(f"[API] AlertCaptureWrapper.add_notification called: {message}")
                        
                        # Skip creating complete alerts in streaming mode - only tokens should be sent
                        if self.server.current_config.streaming_mode:
                            print(f"[API] Skipping alert creation in streaming mode - only tokens are sent")
                            return
                        
                        alert = AlertNotification(
                            id=str(len(self.server.alerts)),
                            timestamp=datetime.now(),
                            message=message,
                            confidence=0.8
                        )
                        self.server.alerts.append(alert)
                        print(f"[API] Alert stored! Total alerts: {len(self.server.alerts)}")
                        
                        # Add to queue for WebSocket broadcasting
                        self.server.pending_alerts_queue.append(alert)
                        print(f"[API] Alert added to broadcast queue. Queue size: {len(self.server.pending_alerts_queue)}")
                        
                        # Also call original if it exists
                        if self.original_web_display and hasattr(self.original_web_display, 'add_notification'):
                            self.original_web_display.add_notification(message)
                    
                    def __getattr__(self, name):
                        # Delegate other methods to original
                        if self.original_web_display:
                            return getattr(self.original_web_display, name)
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
                # Replace the web_display
                original = self.video_system.llm_sink.web_display
                wrapper = AlertCaptureWrapper(self, original)
                self.video_system.llm_sink.web_display = wrapper
                print(f"[API] REPLACED LLM web_display with AlertCaptureWrapper")
                print(f"[API] New web_display type: {type(self.video_system.llm_sink.web_display)}")
            
            # Run processing
            print(f"[API] Starting video processing...")
            
            # Start processing in a separate thread so we can modify the LLM sink after it initializes
            import threading
            import time
            
            def run_with_wrapper_fix():
                # Start the video processing
                self.video_system.run()
            
            # Start video processing in background
            processing_thread = threading.Thread(target=run_with_wrapper_fix, daemon=True)
            processing_thread.start()
            
            # Wait a moment for LLM workers to start, then apply the wrapper
            time.sleep(2)
            
            if self.current_config.mode in ["alert", "realtime_description"] and hasattr(self.video_system, 'llm_sink'):
                print(f"[API] Applying wrapper AFTER LLM workers started...")
                
                # Create a new wrapper
                class DelayedAlertWrapper:
                    def __init__(self, server):
                        self.server = server
                        print(f"[API] DelayedAlertWrapper created")
                    
                    def add_notification(self, message):
                        print(f"[API] DelayedAlertWrapper.add_notification: {message}")
                        
                        # Skip creating complete alerts in streaming mode - only tokens should be sent
                        if hasattr(self.server, 'current_config') and self.server.current_config.streaming_mode:
                            print(f"[API] Skipping alert creation in streaming mode - only tokens are sent")
                            return
                        
                        alert = AlertNotification(
                            id=str(len(self.server.alerts)),
                            timestamp=datetime.now(),
                            message=message,
                            confidence=0.8
                        )
                        self.server.alerts.append(alert)
                        print(f"[API] ALERT STORED! Total: {len(self.server.alerts)}")
                        
                        # Add to queue for WebSocket broadcasting
                        self.server.pending_alerts_queue.append(alert)
                        print(f"[API] Alert added to broadcast queue. Queue size: {len(self.server.pending_alerts_queue)}")
                    
                    def send_streaming_token(self, token):
                        """Handle streaming tokens from Gemini."""
                        print(f"[API] 🎪 DelayedAlertWrapper.send_streaming_token: '{token}'")
                        self.server.pending_token_queue.append({
                            'token': token,
                            'timestamp': time.time()
                        })
                        print(f"[API] ✅ Token added to queue! Queue size: {len(self.server.pending_token_queue)}")
                    
                    def __getattr__(self, name):
                        # For any other attributes, return a dummy function
                        return lambda *args, **kwargs: None
                
                # Force replace the web_display
                wrapper = DelayedAlertWrapper(self)
                self.video_system.llm_sink.web_display = wrapper
                print(f"[API] FORCE REPLACED web_display after LLM start")
                print(f"[API] LLM sink web_display NOW: {type(self.video_system.llm_sink.web_display)}")
            
            # Wait for processing to complete
            processing_thread.join()
            print(f"[API] Video processing completed")
            
        except Exception as e:
            print(f"[API] Processing error: {e}")
        finally:
            self.is_processing = False
            if self.video_system:
                # VideoGatingSystem doesn't have cleanup, so clean up manually
                if hasattr(self.video_system, 'web_display') and self.video_system.web_display:
                    self.video_system.web_display.stop()
                if hasattr(self.video_system, 'llm_sink'):
                    self.video_system.llm_sink.shutdown(wait_for_completion=False, timeout=2)
    
    def run(self, host="0.0.0.0", port=8000):
        """Start the API server."""
        print(f"🚀 Starting Soteira API Server at http://{host}:{port}")
        print(f"📖 API docs available at http://{host}:{port}/docs")
        uvicorn.run(self.app, host=host, port=port)


# Global server instance
server = VideoAnalysisServer()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Soteira Video Analysis API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    server.run(host=args.host, port=args.port)