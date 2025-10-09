from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import base64
import io
from PIL import Image

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
EAR_CONSEC_FRAMES = 3
blink_counter = 0
fatigue_score = 0.0

# ---------- Helper ----------
def smooth(prev, new, alpha=0.2):
    return alpha * new + (1 - alpha) * prev

def analyze_frame(frame_bytes):
    global blink_counter, fatigue_score
    
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(frame_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple face detection simulation
        # In a real scenario, you'd use a proper face detection model
        height, width = img_array.shape[:2]
        
        # Simulate face detection based on image characteristics
        # This is a placeholder - you'd replace with actual ML model
        face_detected = detect_face_simulation(img_array)
        
        if face_detected:
            # Simulate eye detection
            eyes_detected = detect_eyes_simulation(img_array)
            
            if eyes_detected < 2:
                blink_counter += 1
            else:
                blink_counter = 0
                
            score = 1.0 if blink_counter >= EAR_CONSEC_FRAMES else 0.0
            fatigue_score = smooth(fatigue_score, score)
        else:
            blink_counter = 0
            fatigue_score = smooth(fatigue_score, 0.0)
            
        return {
            "fatigue_score": float(fatigue_score),
            "blink_counter": int(blink_counter),
            "eyes_detected": eyes_detected if face_detected else 0,
            "faces_detected": 1 if face_detected else 0,
            "status": "analyzed"
        }
        
    except Exception as e:
        return {
            "fatigue_score": 0.0,
            "blink_counter": 0,
            "error": str(e),
            "status": "error"
        }

def detect_face_simulation(img_array):
    """Simulate face detection - replace with actual model"""
    # Simple brightness-based detection
    avg_brightness = np.mean(img_array)
    return avg_brightness > 50  # Arbitrary threshold

def detect_eyes_simulation(img_array):
    """Simulate eye detection - replace with actual model"""
    # Return random number of eyes for simulation
    import random
    return random.choice([0, 1, 2])

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

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)