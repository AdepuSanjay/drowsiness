from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

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
EYE_CLOSED_CONSEC_FRAMES = 8  # About 2 seconds at 4 FPS
closed_eye_counter = 0
drowsy_alert_triggered = False

# Load Haar cascades - using more reliable cascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def analyze_frame(frame_bytes):
    global closed_eye_counter, drowsy_alert_triggered
    
    try:
        # Convert bytes to image
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                "status": "NO FRAME",
                "is_drowsy": False,
                "alert_sound": False,
                "eyes_detected": 0,
                "closed_eye_counter": 0
            }
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with better parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        eyes_detected = 0
        status = "NO FACE DETECTED"
        is_drowsy = False
        alert_sound = False
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extract face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within face region with better parameters
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                
                eyes_detected = len(eyes)
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw eye rectangles and count valid eyes
                valid_eyes = 0
                for (ex, ey, ew, eh) in eyes:
                    # Filter eyes - they should be in upper half of face
                    if ey < h/2:
                        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                        valid_eyes += 1
                
                eyes_detected = valid_eyes
                
                # DROWSINESS DETECTION LOGIC
                if eyes_detected < 2:  # Less than 2 eyes detected
                    closed_eye_counter += 1
                    status = f"EYES CLOSING ({closed_eye_counter}/{EYE_CLOSED_CONSEC_FRAMES})"
                    
                    if closed_eye_counter >= EYE_CLOSED_CONSEC_FRAMES:
                        status = "DROWSY! ALERT!"
                        is_drowsy = True
                        alert_sound = True
                        drowsy_alert_triggered = True
                    else:
                        alert_sound = False
                        
                else:  # 2 eyes detected - awake
                    if drowsy_alert_triggered:
                        status = "RECOVERING..."
                        alert_sound = False
                        # Reset after a few frames of open eyes
                        if closed_eye_counter == 0:
                            drowsy_alert_triggered = False
                    else:
                        status = "AWAKE"
                    closed_eye_counter = max(0, closed_eye_counter - 2)
                    alert_sound = False
                
                break
        else:
            closed_eye_counter = 0
            status = "NO FACE - MOVE CLOSER"
            alert_sound = False
        
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
        elif "EYES CLOSING" in status:
            # Yellow background for warning
            cv2.rectangle(frame, (0, 0), (width, 60), (0, 255, 255), -1)
            cv2.putText(frame, "⚠️ EYES CLOSING ⚠️", (width//2 - 120, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            # Green background for awake
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 255, 0), -1)
            cv2.putText(frame, f"✅ {status}", (width//2 - 70, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Info at bottom
        cv2.rectangle(frame, (0, height-40), (width, height), (0, 0, 0), -1)
        info_text = f"Faces: {len(faces)} | Eyes: {eyes_detected} | Closed Frames: {closed_eye_counter}"
        cv2.putText(frame, info_text, (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to base64 for frontend
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": status,
            "is_drowsy": is_drowsy,
            "alert_sound": alert_sound,
            "eyes_detected": eyes_detected,
            "closed_eye_counter": closed_eye_counter,
            "faces_detected": len(faces),
            "processed_frame": processed_frame_base64
        }
        
    except Exception as e:
        return {
            "status": f"ERROR: {str(e)}",
            "is_drowsy": False,
            "alert_sound": False,
            "eyes_detected": 0,
            "closed_eye_counter": 0,
            "faces_detected": 0
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
        <title>Drowsiness Detector - FIXED</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Drowsiness Detector - FIXED</h1>
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
                    <div id="status" class="metric-value">AWAKE</div>
                </div>
                <div class="metric-card">
                    <div>Eyes Detected</div>
                    <div id="eyesCount" class="metric-value">0</div>
                </div>
                <div class="metric-card">
                    <div>Closed Frames</div>
                    <div id="closedCounter" class="metric-value">0</div>
                </div>
            </div>
            
            <div class="instructions">
                <strong>How to test:</strong>
                <ul>
                    <li>Make sure your face is clearly visible</li>
                    <li>Good lighting is important</li>
                    <li>Close your eyes for 2+ seconds</li>
                    <li>System will detect and play alert sound</li>
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
            const eyesCountEl = document.getElementById('eyesCount');
            const closedCounterEl = document.getElementById('closedCounter');
            const alertPanel = document.getElementById('alertPanel');
            const alertSound = document.getElementById('alertSound');
            
            let isProcessing = false;
            
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
                    };
                    
                } catch (err) {
                    alert('Camera error: ' + err.message);
                }
            }
            
            // Process frame
            async function processFrame() {
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
                            eyesCountEl.textContent = data.eyes_detected;
                            closedCounterEl.textContent = data.closed_eye_counter;
                            
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
                        }
                    } catch (err) {
                        console.error('Error:', err);
                    }
                    
                    isProcessing = false;
                    requestAnimationFrame(processFrame);
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
    return {"status": "healthy", "message": "Drowsiness detector is working!"}