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
    allow_origins=["*"],  # Fixed: Allow all origins
    allow_credentials=True,
    allow_methods=["*"],   # Fixed: Allow all methods
    allow_headers=["*"],
)

# ---------- Config ----------
EYE_CLOSED_CONSEC_FRAMES = 10
closed_eye_counter = 0
drowsy_frames = 0

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Helper Functions ----------
def analyze_frame(frame_bytes):
    global closed_eye_counter, drowsy_frames
    
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
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eyes_detected = 0
    status = "AWAKE"
    is_drowsy = False
    alert_sound = False
    
    # Create overlay frame
    overlay_frame = frame.copy()
    height, width = overlay_frame.shape[:2]
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(30, 30))
            eyes_detected = len(eyes)
            
            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(overlay_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            # Drowsiness detection
            if len(eyes) < 2:  # Less than 2 eyes detected
                closed_eye_counter += 1
                if closed_eye_counter >= EYE_CLOSED_CONSEC_FRAMES:
                    status = "DROWSY"
                    is_drowsy = True
                    alert_sound = True
                    drowsy_frames += 1
                else:
                    status = "EYES CLOSING"
                    is_drowsy = False
                    alert_sound = False
            else:
                closed_eye_counter = 0
                drowsy_frames = 0
                status = "AWAKE"
                is_drowsy = False
                alert_sound = False
            break
    else:
        closed_eye_counter = 0
        status = "NO FACE"
        is_drowsy = False
        alert_sound = False
    
    # Add status overlay directly on the frame
    if status == "DROWSY":
        # Red background for drowsy
        cv2.rectangle(overlay_frame, (0, 0), (width, 100), (0, 0, 255), -1)
        cv2.putText(overlay_frame, "🚨 DROWSY DETECTED! 🚨", (width//2 - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(overlay_frame, "ALERT! WAKE UP!", (width//2 - 120, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif status == "EYES CLOSING":
        # Yellow background for warning
        cv2.rectangle(overlay_frame, (0, 0), (width, 80), (0, 255, 255), -1)
        cv2.putText(overlay_frame, "⚠️ EYES CLOSING ⚠️", (width//2 - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(overlay_frame, f"Count: {closed_eye_counter}/{EYE_CLOSED_CONSEC_FRAMES}", 
                   (width//2 - 100, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        # Green background for awake
        cv2.rectangle(overlay_frame, (0, 0), (width, 60), (0, 255, 0), -1)
        cv2.putText(overlay_frame, f"✅ {status}", (width//2 - 80, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add info at bottom
    cv2.putText(overlay_frame, f"Eyes: {eyes_detected} | Closed Frames: {closed_eye_counter}", 
               (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert processed frame to base64
    _, buffer = cv2.imencode('.jpg', overlay_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
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
        return JSONResponse({
            "status": "ERROR", 
            "is_drowsy": False,
            "alert_sound": False,
            "error": str(e)
        })

@app.get("/")
async def home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Drowsiness detector is running"}