from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
from io import BytesIO

app = FastAPI(title="Drowsiness Detector (Browser Webcam)")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
EYE_CLOSED_CONSEC_FRAMES = 10  # Number of consecutive frames with closed eyes to trigger alert
ALERT_THRESHOLD = 0.7
closed_eye_counter = 0
fatigue_score = 0.0

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Helper Functions ----------
def smooth(prev, new, alpha=0.3):
    return alpha * new + (1 - alpha) * prev

def analyze_frame(frame_bytes):
    global closed_eye_counter, fatigue_score
    
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {
            "fatigue_score": 0.0,
            "status": "AWAKE",
            "is_drowsy": False,
            "alert_sound": False,
            "eyes_detected": 0
        }
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    score = 0.0
    status = "AWAKE"
    is_drowsy = False
    alert_sound = False
    eyes_detected = 0
    
    # Draw on frame
    overlay_frame = frame.copy()
    height, width = overlay_frame.shape[:2]
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(overlay_frame, "FACE DETECTED", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(30, 30))
            eyes_detected = len(eyes)
            
            # Draw eye rectangles and count
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(overlay_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            # Eye status text
            eye_status = f"EYES: {len(eyes)}"
            cv2.putText(overlay_frame, eye_status, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Drowsiness detection logic
            if len(eyes) < 2:  # Less than 2 eyes detected
                closed_eye_counter += 1
                status = "EYES CLOSING"
                
                if closed_eye_counter >= EYE_CLOSED_CONSEC_FRAMES:
                    status = "DROWSY!"
                    is_drowsy = True
                    alert_sound = True
                    score = 1.0
                else:
                    score = closed_eye_counter / EYE_CLOSED_CONSEC_FRAMES
                    
            else:  # 2 eyes detected - awake
                closed_eye_counter = max(0, closed_eye_counter - 2)
                status = "AWAKE"
                is_drowsy = False
                alert_sound = False
                score = 0.0
            
            fatigue_score = smooth(fatigue_score, score)
            break
    else:
        closed_eye_counter = 0
        fatigue_score = smooth(fatigue_score, 0.0)
        status = "NO FACE DETECTED"
        is_drowsy = False
        alert_sound = False
    
    # Add large status overlay directly on the frame
    if status == "DROWSY!":
        # Red background for drowsy
        cv2.rectangle(overlay_frame, (0, 0), (width, 100), (0, 0, 255), -1)
        cv2.putText(overlay_frame, "🚨 DROWSY DETECTED! 🚨", (width//2 - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(overlay_frame, "ALERT! WAKE UP!", (width//2 - 120, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif status == "EYES CLOSING":
        # Yellow background for warning
        cv2.rectangle(overlay_frame, (0, 0), (width, 80), (0, 255, 255), -1)
        cv2.putText(overlay_frame, "⚠️ EYES CLOSING ⚠️", (width//2 - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(overlay_frame, f"Count: {closed_eye_counter}/{EYE_CLOSED_CONSEC_FRAMES}", 
                   (width//2 - 80, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        # Green background for awake
        cv2.rectangle(overlay_frame, (0, 0), (width, 60), (0, 255, 0), -1)
        cv2.putText(overlay_frame, f"✅ {status}", (width//2 - 100, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add fatigue score and eye counter at bottom
    cv2.rectangle(overlay_frame, (0, height-40), (width, height), (0, 0, 0), -1)
    cv2.putText(overray_frame, f"Fatigue: {fatigue_score:.2f} | Eyes: {eyes_detected} | Closed Frames: {closed_eye_counter}", 
               (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert processed frame to base64 for sending to frontend
    _, buffer = cv2.imencode('.jpg', overlay_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "fatigue_score": float(fatigue_score),
        "status": status,
        "is_drowsy": is_drowsy,
        "alert_sound": alert_sound,
        "eyes_detected": eyes_detected,
        "closed_eye_counter": closed_eye_counter,
        "processed_frame": processed_frame_base64
    }

# ---------- Routes ----------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        frame_bytes = await file.read()
        result = analyze_frame(frame_bytes)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "fatigue_score": 0.0, "status": "ERROR", "alert_sound": False})

@app.get("/")
async def home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {"status": "healthy"}