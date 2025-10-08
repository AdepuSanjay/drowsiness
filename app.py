from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import time

app = FastAPI(title="Drowsiness Detector")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
EYE_CLOSED_CONSEC_FRAMES = 10  # About 2-3 seconds
closed_eye_counter = 0
last_eye_state = "open"
last_update_time = time.time()

# Load only face cascade (more reliable)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_frame(frame_bytes):
    global closed_eye_counter, last_eye_state, last_update_time
    
    try:
        # Convert bytes to image
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                "status": "NO FRAME",
                "is_drowsy": False,
                "alert_sound": False,
                "eye_state": "unknown",
                "closed_eye_counter": 0
            }
        
        # Resize for consistency
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        current_time = time.time()
        status = "NO FACE - MOVE CLOSER"
        is_drowsy = False
        alert_sound = False
        eye_state = "no_face"
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # SIMPLIFIED APPROACH: Use manual eye regions
                # Estimate eye regions in the face
                eye_region_height = h // 3
                eye_region_width = w // 2
                
                # Left eye region (top-left of face)
                left_eye_x = x + w//4
                left_eye_y = y + h//4
                left_eye_roi = gray[left_eye_y:left_eye_y+eye_region_height, 
                                   left_eye_x:left_eye_x+eye_region_width]
                
                # Right eye region (top-right of face)  
                right_eye_x = x + w//2
                right_eye_y = y + h//4
                right_eye_roi = gray[right_eye_y:right_eye_y+eye_region_height, 
                                    right_eye_x:right_eye_x+eye_region_width]
                
                # Draw eye regions
                cv2.rectangle(frame, (left_eye_x, left_eye_y), 
                            (left_eye_x+eye_region_width, left_eye_y+eye_region_height), 
                            (0, 255, 0), 1)
                cv2.rectangle(frame, (right_eye_x, right_eye_y), 
                            (right_eye_x+eye_region_width, right_eye_y+eye_region_height), 
                            (0, 255, 0), 1)
                
                # SIMPLE EYE STATE DETECTION USING BRIGHTNESS
                # When eyes are open, eye regions have more variation (pupils, whites)
                # When eyes are closed, eye regions are more uniform (eyelids)
                
                left_eye_brightness = np.mean(left_eye_roi) if left_eye_roi.size > 0 else 0
                right_eye_brightness = np.mean(right_eye_roi) if right_eye_roi.size > 0 else 0
                
                # Calculate brightness variation (simple proxy for eye state)
                left_eye_std = np.std(left_eye_roi) if left_eye_roi.size > 0 else 0
                right_eye_std = np.std(right_eye_roi) if right_eye_roi.size > 0 else 0
                
                avg_std = (left_eye_std + right_eye_std) / 2
                
                # Determine eye state based on variation
                if avg_std > 25:  # High variation = eyes open
                    eye_state = "open"
                    closed_eye_counter = max(0, closed_eye_counter - 1)
                    status = "AWAKE - EYES OPEN"
                else:  # Low variation = eyes likely closed
                    eye_state = "closed"
                    closed_eye_counter += 1
                    status = f"EYES CLOSED ({closed_eye_counter}/{EYE_CLOSED_CONSEC_FRAMES})"
                
                # Drowsiness detection
                if closed_eye_counter >= EYE_CLOSED_CONSEC_FRAMES:
                    status = "DROWSY! ALERT!"
                    is_drowsy = True
                    alert_sound = True
                
                # Add eye state info to frame
                cv2.putText(frame, f"Eye State: {eye_state}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Variation: {avg_std:.1f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                break
        
        # ADD OVERLAY TO FRAME
        height, width = frame.shape[:2]
        
        # Status banner at top
        if "DROWSY" in status:
            # Red background for drowsy
            cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 255), -1)
            cv2.putText(frame, "🚨 DROWSY DETECTED! 🚨", (width//2 - 180, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "WAKE UP! ALERT! WAKE UP!", (width//2 - 150, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif "EYES CLOSED" in status:
            # Yellow background for warning
            cv2.rectangle(frame, (0, 0), (width, 60), (0, 255, 255), -1)
            cv2.putText(frame, "⚠️ EYES CLOSED ⚠️", (width//2 - 100, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            # Green background for awake
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 255, 0), -1)
            cv2.putText(frame, f"✅ {status}", (width//2 - 70, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Info at bottom
        cv2.rectangle(frame, (0, height-40), (width, height), (0, 0, 0), -1)
        info_text = f"Faces: {len(faces)} | Eye State: {eye_state} | Closed Frames: {closed_eye_counter}"
        cv2.putText(frame, info_text, (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to base64 for frontend
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": status,
            "is_drowsy": is_drowsy,
            "alert_sound": alert_sound,
            "eye_state": eye_state,
            "closed_eye_counter": closed_eye_counter,
            "faces_detected": len(faces),
            "processed_frame": processed_frame_base64
        }
        
    except Exception as e:
        return {
            "status": f"ERROR",
            "is_drowsy": False,
            "alert_sound": False,
            "eye_state": "error",
            "closed_eye_counter": 0
        }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        frame_bytes = await file.read()
        result = analyze_frame(frame_bytes)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "SERVER ERROR", 
            "is_drowsy": False,
            "alert_sound": False
        })

@app.get("/")
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drowsiness Detector - SIMPLIFIED</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                max-width: 800px;
                width: 100%;
                text-align: center;
            }
            h1 { color: #333; margin-bottom: 10px; }
            .subtitle { color: #666; margin-bottom: 20px; }
            .camera-container {
                position: relative;
                margin: 20px auto;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                max-width: 640px;
            }
            #video { width: 100%; display: block; }
            #processedCanvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }
            .alert-panel {
                background: linear-gradient(45deg, #ff416c, #ff4b2b);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
                animation: alert-pulse 0.5s infinite alternate;
            }
            .alert-panel.show { display: block; }
            @keyframes alert-pulse {
                0% { transform: scale(1); opacity: 1; }
                100% { transform: scale(1.02); opacity: 0.9; }
            }
            .instructions {
                background: #e9ecef;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: left;
            }
            .debug-info {
                background: #333;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Drowsiness Detector - SIMPLIFIED</h1>
            <p class="subtitle">Close your eyes for 2+ seconds to test</p>
            
            <div class="camera-container">
                <video id="video" autoplay playsinline></video>
                <canvas id="processedCanvas"></canvas>
            </div>
            
            <div id="alertPanel" class="alert-panel">
                🚨 ALERT: DROWSINESS DETECTED! 🚨
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div>Status</div>
                    <div id="status" class="metric-value">LOADING...</div>
                </div>
                <div class="metric-card">
                    <div>Eye State</div>
                    <div id="eyeState" class="metric-value">-</div>
                </div>
                <div class="metric-card">
                    <div>Closed Frames</div>
                    <div id="closedCounter" class="metric-value">0</div>
                </div>
            </div>

            <div class="debug-info">
                Debug: <span id="debugInfo">Waiting for camera...</span>
            </div>
            
            <div class="instructions">
                <strong>How it works now:</strong>
                <ul>
                    <li>System detects your FACE (blue rectangle)</li>
                    <li>Estimates EYE regions (green rectangles)</li>
                    <li>Measures brightness variation in eye areas</li>
                    <li>High variation = Eyes OPEN | Low variation = Eyes CLOSED</li>
                    <li>Close eyes for 2+ seconds to trigger alert</li>
                </ul>
            </div>
            
            <audio id="alertSound" preload="auto">
                <source src="https://assets.mixkit.co/active_storage/sfx/259/259-preview.mp3" type="audio/mpeg">
            </audio>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('processedCanvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('status');
            const eyeStateEl = document.getElementById('eyeState');
            const closedCounterEl = document.getElementById('closedCounter');
            const alertPanel = document.getElementById('alertPanel');
            const alertSound = document.getElementById('alertSound');
            const debugInfo = document.getElementById('debugInfo');
            
            let isProcessing = false;
            let frameCount = 0;
            
            // Set canvas size
            canvas.width = 640;
            canvas.height = 480;
            
            // Access webcam
            async function setupCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: { ideal: 640 },
                            height: { ideal: 480 },
                            facingMode: 'user'
                        } 
                    });
                    video.srcObject = stream;
                    
                    video.onloadedmetadata = () => {
                        console.log('Camera ready');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        debugInfo.textContent = 'Camera ready - Processing frames...';
                    };
                    
                } catch (err) {
                    debugInfo.textContent = 'Camera error: ' + err.message;
                    alert('Camera error: ' + err.message);
                }
            }
            
            // Process frame
            async function processFrame() {
                frameCount++;
                
                if (isProcessing || video.readyState !== video.HAVE_ENOUGH_DATA) {
                    requestAnimationFrame(processFrame);
                    return;
                }
                
                isProcessing = true;
                
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCtx.drawImage(video, 0, 0);
                
                tempCanvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            
                            // Update UI
                            statusEl.textContent = data.status;
                            eyeStateEl.textContent = data.eye_state;
                            closedCounterEl.textContent = data.closed_eye_counter;
                            debugInfo.textContent = `Frames: ${frameCount} | Faces: ${data.faces_detected} | State: ${data.eye_state}`;
                            
                            // Color code eye state
                            if (data.eye_state === 'open') {
                                eyeStateEl.style.color = '#4CAF50';
                            } else if (data.eye_state === 'closed') {
                                eyeStateEl.style.color = '#ff9800';
                            } else {
                                eyeStateEl.style.color = '#666';
                            }
                            
                            // Handle alerts
                            if (data.alert_sound) {
                                alertPanel.classList.add('show');
                                alertSound.play().catch(e => console.log('Audio play failed'));
                            } else {
                                alertPanel.classList.remove('show');
                            }
                            
                            // Display processed frame
                            if (data.processed_frame) {
                                const img = new Image();
                                img.onload = () => {
                                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                                };
                                img.src = 'data:image/jpeg;base64,' + data.processed_frame;
                            }
                        } else {
                            debugInfo.textContent = 'Server error: ' + response.status;
                        }
                    } catch (err) {
                        debugInfo.textContent = 'Fetch error: ' + err.message;
                        console.error('Error:', err);
                    }
                    
                    isProcessing = false;
                    setTimeout(() => requestAnimationFrame(processFrame), 100); // ~10 FPS
                }, 'image/jpeg', 0.7);
            }
            
            // Initialize
            setupCamera().then(() => {
                setTimeout(() => processFrame(), 1000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Simplified drowsiness detector is working!"}