from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os

app = FastAPI(title="Drowsiness Detector (Browser Webcam)")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
)

# ---------- Static Files ----------
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Config ----------
EAR_CONSEC_FRAMES = 3  # Reduced for testing
ALERT_THRESHOLD = 0.75
blink_counter = 0
fatigue_score = 0.0

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Helper ----------
def smooth(prev, new, alpha=0.2):
    return alpha * new + (1 - alpha) * prev

def analyze_frame(frame_bytes):
    global blink_counter, fatigue_score
    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode image"}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        score = 0.0
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Debug: print number of eyes detected
                print(f"Eyes detected: {len(eyes)}")
                
                if len(eyes) < 2:  # Less than 2 eyes detected
                    blink_counter += 1
                    print(f"Eyes closed frame: {blink_counter}")
                else:
                    blink_counter = 0
                
                # If eyes closed for consecutive frames, mark as drowsy
                if blink_counter >= EAR_CONSEC_FRAMES:
                    score = 1.0
                    print("ALERT: Drowsiness detected!")
                else:
                    score = 0.0
                    
                fatigue_score = smooth(fatigue_score, score)
                break
        else:
            blink_counter = 0
            fatigue_score = smooth(fatigue_score, 0.0)
            
        return {
            "fatigue_score": float(fatigue_score),
            "blink_counter": int(blink_counter),
            "eyes_detected": len(eyes) if 'eyes' in locals() else 0,
            "faces_detected": len(faces)
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- Routes ----------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    frame_bytes = await file.read()
    result = analyze_frame(frame_bytes)
    return JSONResponse(result)

@app.get("/")
async def home():
    # Serve the HTML file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drowsiness Detector</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .camera-container { position: relative; }
            #video, #canvas { border: 2px solid #333; margin: 10px 0; }
            .alert { color: red; font-weight: bold; font-size: 24px; }
            .normal { color: green; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Drowsiness Detector</h1>
            <div class="camera-container">
                <video id="video" width="640" height="480" autoplay></video>
                <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
            </div>
            <div>
                <h2>Fatigue Score: <span id="fatigue" class="normal">0</span></h2>
                <p>Blink Counter: <span id="blinkCounter">0</span></p>
                <p>Eyes Detected: <span id="eyesDetected">0</span></p>
                <p>Status: <span id="status" class="normal">Normal</span></p>
            </div>
            <audio id="alertSound" src="https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3" preload="auto"></audio>
        </div>

        <script>
            const video = document.getElementById("video");
            const canvas = document.getElementById("canvas");
            const ctx = canvas.getContext("2d");
            const fatigueEl = document.getElementById("fatigue");
            const blinkCounterEl = document.getElementById("blinkCounter");
            const eyesDetectedEl = document.getElementById("eyesDetected");
            const statusEl = document.getElementById("status");
            const alertSound = document.getElementById("alertSound");
            
            const ALERT_THRESHOLD = 0.75;

            // Access webcam
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                })
                .catch(err => {
                    alert("Error accessing webcam: " + err);
                });

            // Capture frame and send to server every 500ms
            setInterval(async () => {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append("file", blob, "frame.jpg");
                    
                    try {
                        // Use relative path for Vercel deployment
                        const response = await fetch("/analyze", {
                            method: "POST",
                            body: formData
                        });
                        const data = await response.json();
                        
                        if (data.error) {
                            console.error("Server error:", data.error);
                            return;
                        }
                        
                        fatigueEl.textContent = data.fatigue_score.toFixed(2);
                        blinkCounterEl.textContent = data.blink_counter;
                        eyesDetectedEl.textContent = data.eyes_detected || 0;
                        
                        if (data.fatigue_score > ALERT_THRESHOLD) {
                            fatigueEl.className = "alert";
                            statusEl.textContent = "DROWSY!";
                            statusEl.className = "alert";
                            alertSound.play();
                        } else {
                            fatigueEl.className = "normal";
                            statusEl.textContent = "Normal";
                            statusEl.className = "normal";
                        }
                    } catch(err) {
                        console.error("Error analyzing frame:", err);
                    }
                }, "image/jpeg");
            }, 500);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)