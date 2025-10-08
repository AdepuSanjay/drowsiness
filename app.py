from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
EYE_CLOSED_CONSEC_FRAMES = 5  # Reduced for testing
closed_eye_counter = 0

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def analyze_frame(frame_bytes):
    global closed_eye_counter
    
    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                "status": "NO FRAME",
                "is_drowsy": False,
                "alert_sound": False,
                "eyes_detected": 0,
                "closed_eye_counter": closed_eye_counter,
                "debug": "Frame decoding failed"
            }
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        logger.debug(f"Faces detected: {len(faces)}")
        
        eyes_detected = 0
        status = "NO FACE"
        is_drowsy = False
        alert_sound = False
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(20, 20))
                eyes_detected = len(eyes)
                
                logger.debug(f"Eyes detected in face: {eyes_detected}")
                
                # Simple drowsiness logic
                if eyes_detected < 2:
                    closed_eye_counter += 1
                    status = f"EYES CLOSING ({closed_eye_counter}/{EYE_CLOSED_CONSEC_FRAMES})"
                    
                    if closed_eye_counter >= EYE_CLOSED_CONSEC_FRAMES:
                        status = "DROWSY!"
                        is_drowsy = True
                        alert_sound = True
                else:
                    closed_eye_counter = 0
                    status = "AWAKE"
                    is_drowsy = False
                    alert_sound = False
                
                break
        else:
            closed_eye_counter = 0
            status = "NO FACE"
        
        # Add debug text to frame
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Status overlay
        if status == "DROWSY!":
            cv2.rectangle(overlay_frame, (0, 0), (width, 80), (0, 0, 255), -1)
            cv2.putText(overlay_frame, "🚨 DROWSY! ALERT! 🚨", (50, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif "EYES CLOSING" in status:
            cv2.rectangle(overlay_frame, (0, 0), (width, 60), (0, 255, 255), -1)
            cv2.putText(overlay_frame, status, (50, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(overlay_frame, (0, 0), (width, 50), (0, 255, 0), -1)
            cv2.putText(overlay_frame, f"✅ {status}", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Debug info at bottom
        cv2.putText(overlay_frame, f"Faces: {len(faces)} | Eyes: {eyes_detected}", 
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', overlay_frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": status,
            "is_drowsy": is_drowsy,
            "alert_sound": alert_sound,
            "eyes_detected": eyes_detected,
            "closed_eye_counter": closed_eye_counter,
            "faces_detected": len(faces),
            "processed_frame": processed_frame_base64,
            "debug": f"Faces: {len(faces)}, Eyes: {eyes_detected}"
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_frame: {str(e)}")
        return {
            "status": "ERROR",
            "is_drowsy": False,
            "alert_sound": False,
            "eyes_detected": 0,
            "closed_eye_counter": 0,
            "debug": f"Error: {str(e)}"
        }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        logger.debug("Received analyze request")
        frame_bytes = await file.read()
        logger.debug(f"Frame size: {len(frame_bytes)} bytes")
        result = analyze_frame(frame_bytes)
        logger.debug(f"Analysis result: {result}")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return JSONResponse({
            "status": "SERVER ERROR", 
            "is_drowsy": False,
            "alert_sound": False,
            "debug": f"Server error: {str(e)}"
        })

@app.get("/")
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drowsiness Detector - DEBUG</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .camera-container { position: relative; margin: 20px 0; }
            #video, #processedCanvas { width: 100%; max-width: 640px; border: 2px solid #333; }
            #processedCanvas { position: absolute; top: 0; left: 0; }
            .debug { background: #eee; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .alert { background: red; color: white; padding: 20px; font-size: 24px; text-align: center; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Drowsiness Detector - DEBUG</h1>
            <p>Close your eyes for 3+ seconds to test</p>
            
            <div class="camera-container">
                <video id="video" autoplay playsinline></video>
                <canvas id="processedCanvas"></canvas>
            </div>
            
            <div id="alert" class="alert">🚨 DROWSY DETECTED! 🚨</div>
            
            <div class="debug">
                <h3>Debug Info:</h3>
                <div>Status: <span id="status">Loading...</span></div>
                <div>Faces: <span id="faces">0</span></div>
                <div>Eyes: <span id="eyes">0</span></div>
                <div>Closed Counter: <span id="closedCounter">0</span></div>
                <div>Alert Sound: <span id="alertSound">No</span></div>
                <div>Last Error: <span id="error">None</span></div>
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
            const facesEl = document.getElementById('faces');
            const eyesEl = document.getElementById('eyes');
            const closedCounterEl = document.getElementById('closedCounter');
            const alertSoundEl = document.getElementById('alertSound');
            const alertEl = document.getElementById('alert');
            const alertSoundFlag = document.getElementById('alertSound');
            const errorEl = document.getElementById('error');

            // Set canvas size
            canvas.width = 640;
            canvas.height = 480;

            // Access webcam
            async function setupCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 } 
                    });
                    video.srcObject = stream;
                    console.log('Camera access successful');
                } catch (err) {
                    errorEl.textContent = 'Camera error: ' + err.message;
                    console.error('Camera error:', err);
                }
            }

            // Process frames
            async function processFrame() {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
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
                                
                                // Update debug info
                                statusEl.textContent = data.status;
                                facesEl.textContent = data.faces_detected || 0;
                                eyesEl.textContent = data.eyes_detected || 0;
                                closedCounterEl.textContent = data.closed_eye_counter || 0;
                                alertSoundFlag.textContent = data.alert_sound ? 'YES' : 'NO';
                                errorEl.textContent = data.debug || 'No error';
                                
                                // Show/hide alert
                                if (data.alert_sound) {
                                    alertEl.style.display = 'block';
                                    alertSoundEl.play().catch(e => console.log('Audio error:', e));
                                } else {
                                    alertEl.style.display = 'none';
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
                                errorEl.textContent = 'Server response error: ' + response.status;
                            }
                        } catch (err) {
                            errorEl.textContent = 'Fetch error: ' + err.message;
                            console.error('Fetch error:', err);
                        }
                    }, 'image/jpeg');
                }
                
                setTimeout(processFrame, 200); // Process every 200ms
            }

            // Initialize
            setupCamera().then(() => {
                setTimeout(processFrame, 1000); // Start after 1 second
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Drowsiness detector is running"}