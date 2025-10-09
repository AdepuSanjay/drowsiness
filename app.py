from fastapi import FastAPI, UploadFile, File, Request
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Static Files ----------
# Create static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

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
            return {"fatigue_score": 0.0, "blink_counter": 0, "error": "Invalid image"}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        score = 0.0
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) < 2:
                    blink_counter += 1
                else:
                    blink_counter = 0
                    
                score = 1.0 if blink_counter >= EAR_CONSEC_FRAMES else 0.0
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
        return {"fatigue_score": 0.0, "blink_counter": 0, "error": str(e)}

# ---------- Routes ----------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    frame_bytes = await file.read()
    result = analyze_frame(frame_bytes)
    return JSONResponse(result)

@app.get("/")
async def home():
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Drowsiness Detector</h1><p>index.html not found</p>")

# For Vercel deployment
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)