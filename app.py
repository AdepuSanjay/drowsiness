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
EYE_CLOSED_CONSEC_FRAMES = 8  # Reduced for faster testing
closed_eye_counter = 0

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------- Helper Functions ----------
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
                "closed_eye_counter": 0,
                "faces_detected": 0
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
        status = "NO FACE"
        is_drowsy = False
        alert_sound = False
        
        # Create overlay frame
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(overlay_frame, "FACE", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Region of interest for eyes (upper half of face)
                roi_gray = gray[y:y+int(h/2), x:x+w]
                roi_color = overlay_frame[y:y+int(h/2), x:x+w]
                
                # Detect eyes with better parameters
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20),
                    maxSize=(80, 80)
                )
                
                eyes_detected = len(eyes)
                
                # Draw eye rectangles and count them
                eye_count = 0
                for (ex, ey, ew, eh) in eyes:
                    if eye_count < 2:  # Only draw first 2 eyes
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                        cv2.putText(roi_color, f"EYE {eye_count+1}", (ex, ey-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        eye_count += 1
                
                # Drowsiness detection logic
                if eyes_detected < 2:  # Less than 2 eyes detected
                    closed_eye_counter += 1
                    if closed_eye_counter >= EYE_CLOSED_CONSEC_FRAMES:
                        status = "DROWSY"
                        is_drowsy = True
                        alert_sound = True
                    else:
                        status = f"EYES CLOSING ({closed_eye_counter}/{EYE_CLOSED_CONSEC_FRAMES})"
                        is_drowsy = False
                        alert_sound = False
                else:
                    closed_eye_counter = 0
                    status = "AWAKE"
                    is_drowsy = False
                    alert_sound = False
                
                break
        else:
            closed_eye_counter = 0
            status = "NO FACE DETECTED"
            is_drowsy = False
            alert_sound = False
        
        # Add status overlay directly on the frame
        if status == "DROWSY":
            # Red background for drowsy
            cv2.rectangle(overlay_frame, (0, 0), (width, 100), (0, 0, 255), -1)
            cv2.putText(overlay_frame, "🚨 DROWSY DETECTED! 🚨", (width//2 - 180, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay_frame, "ALERT! WAKE UP!", (width//2 - 100, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif "EYES CLOSING" in status:
            # Yellow background for warning
            cv2.rectangle(overlay_frame, (0, 0), (width, 80), (0, 255, 255), -1)
            cv2.putText(overlay_frame, "⚠️ EYES CLOSING ⚠️", (width//2 - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay_frame, status.split('(')[1].split(')')[0], (width//2 - 60, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            # Green background for awake
            cv2.rectangle(overlay_frame, (0, 0), (width, 60), (0, 255, 0), -1)
            cv2.putText(overlay_frame, f"✅ {status}", (width//2 - 60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add debug info at bottom
        cv2.rectangle(overlay_frame, (0, height-30), (width, height), (0, 0, 0), -1)
        cv2.putText(overlay_frame, f"Faces: {len(faces)} | Eyes: {eyes_detected} | Closed: {closed_eye_counter}", 
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
            "faces_detected": 0,
            "processed_frame": None
        }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        frame_bytes = await file.read()
        result = analyze_frame(frame_bytes)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": f"SERVER ERROR: {str(e)}", 
            "is_drowsy": False,
            "alert_sound": False
        })

@app.get("/")
async def home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "healthy"}