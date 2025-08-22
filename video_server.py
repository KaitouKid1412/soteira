import asyncio
import json
import os
import signal
import ssl
from aiohttp import web
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from phone_stream_server import get_stream_server

pcs = set()  # track active PeerConnections
stream_server = get_stream_server()  # Global stream server instance

INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Phone → Mac WebRTC</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 16px; }
      video { width: 100%; max-width: 480px; border-radius: 12px; }
      button { padding: 10px 16px; border-radius: 10px; border: 1px solid #ccc; }
      .row { display:flex; gap:12px; align-items:center; }
      .ok { color: #2a8f2a; }
      .err { color: #b00020; }
      #ip { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    </style>
  </head>
  <body>
    <h2>Phone → Mac WebRTC</h2>
    <p>1) Ensure phone & Mac are on the <b>same Wi-Fi</b>.<br>
       2) Tap <b>Start</b> to stream your camera to the Mac.</p>
    <div class="row">
      <button id="startBtn">Start</button>
      <span id="status"></span>
    </div>
    <p>Local preview:</p>
    <video id="preview" autoplay playsinline muted></video>

    <script>
      const status = (msg, ok=true) => {
        const el = document.getElementById('status');
        el.textContent = msg;
        el.className = ok ? 'ok' : 'err';
      };

      async function start() {
        try {
          status('Requesting camera…');
          
          // Try modern getUserMedia first, fallback to legacy
          let getUserMedia = null;
          if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
          } else if (navigator.getUserMedia) {
            getUserMedia = navigator.getUserMedia.bind(navigator);
          } else if (navigator.webkitGetUserMedia) {
            getUserMedia = navigator.webkitGetUserMedia.bind(navigator);
          } else if (navigator.mozGetUserMedia) {
            getUserMedia = navigator.mozGetUserMedia.bind(navigator);
          }
          
          if (!getUserMedia) {
            status('Camera access not supported by this browser', false);
            return;
          }
          
          // First, try to get device capabilities
          let stream;
          let deviceCapabilities = null;
          
          try {
            // Get basic stream to check capabilities
            const testStream = await getUserMedia({
              video: { facingMode: 'environment' }
            });
            const testTrack = testStream.getVideoTracks()[0];
            deviceCapabilities = testTrack.getCapabilities();
            console.log('Device capabilities detected:', deviceCapabilities);
            testTrack.stop(); // Release the test stream
            testStream.getTracks().forEach(track => track.stop());
          } catch (e) {
            console.log('Could not detect capabilities:', e);
          }
          
          // Build constraints based on capabilities or use defaults
          const constraints = [];
          
          if (deviceCapabilities && deviceCapabilities.width && deviceCapabilities.height) {
            // Handle different capability formats
            let maxWidth = 1920;
            let maxHeight = 1080;
            
            if (deviceCapabilities.width.max) {
              maxWidth = Array.isArray(deviceCapabilities.width.max) ? 
                Math.max(...deviceCapabilities.width.max) : deviceCapabilities.width.max;
            }
            if (deviceCapabilities.height.max) {
              maxHeight = Array.isArray(deviceCapabilities.height.max) ? 
                Math.max(...deviceCapabilities.height.max) : deviceCapabilities.height.max;
            }
            
            console.log(`Max supported: ${maxWidth}x${maxHeight}`);
            
            // Try maximum supported resolution with more flexible constraints
            constraints.push({
              video: {
                facingMode: 'environment',
                width: { ideal: maxWidth, max: maxWidth },
                height: { ideal: maxHeight, max: maxHeight },
                frameRate: { ideal: 30 }
              },
              audio: false
            });
          }
          
          // Add progressive fallback constraints
          constraints.push(
            // Very high quality with ideal constraints
            {
              video: { 
                facingMode: 'environment',
                width: { ideal: 1920, min: 1280 },
                height: { ideal: 1080, min: 720 },
                frameRate: { ideal: 30 }
              },
              audio: false
            },
            // High quality with ideal constraints
            {
              video: { 
                facingMode: 'environment',
                width: { ideal: 1280, min: 960 },
                height: { ideal: 720, min: 540 },
                frameRate: { ideal: 30 }
              },
              audio: false
            },
            // Medium quality with ideal constraints
            {
              video: { 
                facingMode: 'environment',
                width: { ideal: 960, min: 640 },
                height: { ideal: 540, min: 480 },
                frameRate: { ideal: 30 }
              },
              audio: false
            },
            // Basic with minimum requirements
            {
              video: { 
                facingMode: 'environment',
                width: { min: 640 },
                height: { min: 480 }
              },
              audio: false
            },
            // Absolute fallback
            {
              video: { facingMode: 'environment' },
              audio: false
            }
          );
          
          for (let i = 0; i < constraints.length; i++) {
            try {
              console.log(`Trying constraints level ${i + 1}:`, constraints[i].video);
              stream = await getUserMedia(constraints[i]);
              console.log(`Video constraints succeeded at level ${i + 1}`);
              
              // Show which level was successful in the status
              status(`Camera ready (level ${i + 1})`);
              break;
            } catch (e) {
              console.log(`Video constraints level ${i + 1} failed:`, e.message);
              status(`Level ${i + 1} failed, trying next...`);
              if (i === constraints.length - 1) throw e;
            }
          }
          
          // Log actual video track settings
          const videoTrack = stream.getVideoTracks()[0];
          const settings = videoTrack.getSettings();
          console.log('Video track settings:', settings);
          
          // Log capabilities to see what your phone actually supports
          const capabilities = videoTrack.getCapabilities();
          console.log('Video track capabilities:', capabilities);
          
          status(`Camera: ${settings.width}x${settings.height} @ ${settings.frameRate}fps`);
          document.getElementById('preview').srcObject = stream;

          const pc = new RTCPeerConnection({
            iceServers: [] // LAN use; add STUN if needed
          });

          // Send phone camera to Mac with quality settings
          for (const track of stream.getTracks()) {
            const sender = pc.addTrack(track, stream);
            
            // Set encoding parameters for better quality
            if (track.kind === 'video') {
              try {
                const params = sender.getParameters();
                if (params.encodings && params.encodings.length > 0) {
                  // Conservative quality settings to avoid compatibility issues
                  params.encodings[0].maxBitrate = 5000000; // 5 Mbps
                  params.encodings[0].maxFramerate = 30;
                  params.encodings[0].scaleResolutionDownBy = 1; // No downscaling
                  
                  await sender.setParameters(params);
                  console.log('Set encoding parameters successfully');
                }
              } catch (e) {
                console.log('Failed to set encoding parameters:', e);
                // Continue anyway - basic WebRTC will still work
              }
            }
          }

          // Optional: log ICE gathering for debug
          pc.addEventListener('icegatheringstatechange', () => {
            console.log('icegatheringstate:', pc.iceGatheringState);
          });

          status('Creating offer…');
          const offer = await pc.createOffer({
            offerToReceiveAudio: false,
            offerToReceiveVideo: false,
            // Enforce high quality video
            voiceActivityDetection: false,
            iceRestart: false
          });
          
          // Log original SDP for debugging
          console.log('Original SDP length:', offer.sdp.length);
          
          // Don't modify SDP - causes parsing errors
          // Just rely on encoding parameters instead
          await pc.setLocalDescription(offer);

          // Post offer to server, get answer
          status('Sending offer…');
          const resp = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              sdp: pc.localDescription.sdp,
              type: pc.localDescription.type
            })
          });

          if (!resp.ok) {
            status('Server /offer failed (' + resp.status + ')', false);
            return;
          }
          const answer = await resp.json();
          await pc.setRemoteDescription(answer);

          status('Streaming! Keep this page open.');
        } catch (e) {
          console.error(e);
          status('Error: ' + e, false);
        }
      }

      document.getElementById('startBtn').addEventListener('click', start);
    </script>
  </body>
</html>
"""

async def index(request):
    return web.Response(text=INDEX_HTML, content_type="text/html")

async def offer(request: web.Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Local-only ICE (same Wi-Fi). Add STUN for broader networks:
    # config = RTCConfiguration([RTCIceServer("stun:stun.l.google.com:19302")])
    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    print("Created RTCPeerConnection")

    @pc.on("connectionstatechange")
    def on_connectionstatechange():
        print("Connection state:", pc.connectionState)
        if pc.connectionState == "failed":
            print("Connection failed, attempting restart...")
            # Don't immediately close on failed state - allow for recovery
        elif pc.connectionState in ("closed", "disconnected"):
            print("Connection closed/disconnected, cleaning up...")
            asyncio.ensure_future(close_pc(pc))

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)

        if track.kind == "video":
            # Use a single frame handler that serves both display and HTTP
            asyncio.ensure_future(unified_frame_handler(track))

        @track.on("ended")
        async def on_ended():
            print("Track ended:", track.kind)

    # Set remote description (phone offer), create and send answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    
    # Don't modify SDP - can cause parsing errors
    # Quality control is handled via encoding parameters instead
    print(f"Answer SDP length: {len(answer.sdp)}")
    await pc.setLocalDescription(answer)

    # Wait for ICE gathering to complete so the answer includes candidates
    await ice_gathering_complete(pc)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

async def unified_frame_handler(track):
    """
    Unified frame handler that receives frames and sends to stream server.
    No local display for better performance and to avoid OpenCV GUI issues.
    """
    from av import VideoFrame
    
    print("[UNIFIED] Starting unified frame handler (no local display)")

    frame_count = 0
    try:
        while True:
            frame: VideoFrame = await track.recv()
            frame_count += 1
            
            # Log resolution details on first frame and every 100 frames
            if frame_count == 1 or frame_count % 100 == 0:
                print(f"[UNIFIED] Frame #{frame_count}: {frame.width}x{frame.height}, format: {frame.format.name}")
            
            img = frame.to_ndarray(format="bgr24")
            
            # Send to stream server (this feeds the web frontend)
            stream_server.receive_frame(img)
                
    except Exception as e:
        print(f"[UNIFIED] Frame handler ended: {e}")

async def display_frames(track):
    """
    Legacy function - now handled by unified_frame_handler
    """
    pass

async def ice_gathering_complete(pc: RTCPeerConnection):
    """
    Await ICE gathering to 'complete' so we return an answer with candidates.
    """
    if pc.iceGatheringState == "complete":
        return
    fut = asyncio.get_event_loop().create_future()

    @pc.on("icegatheringstatechange")
    def on_state_change():
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(True)

    await asyncio.wait_for(fut, timeout=5.0)  # small timeout is fine on LAN

async def close_pc(pc: RTCPeerConnection):
    if pc in pcs:
        pcs.discard(pc)
    try:
        await pc.close()
    except Exception:
        pass

def create_app():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    
    # Add stream server routes
    stream_app = stream_server.get_app()
    app.add_subapp('/stream/', stream_app)
    
    return app

if __name__ == "__main__":
    app = create_app()

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(app.shutdown()))

    # Create SSL context for HTTPS
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')

    print("Starting HTTPS server...")
    print("Access from your phone at: https://[YOUR_MAC_IP]:8443")
    print("(Accept the security warning for self-signed certificate)")
    
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=ssl_context)
