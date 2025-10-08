from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

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
EAR_CONSEC_FRAMES = 10
blink_counter = 0
fatigue_score = 0.0
drowsy_frames = 0

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ---------- Helper Functions ----------
def smooth(prev, new, alpha=0.3):
    return alpha * new + (1 - alpha) * prev


def analyze_frame(frame_bytes):
    global blink_counter, fatigue_score, drowsy_frames

    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "fatigue_score": 0.0,
            "status": "AWAKE",
            "is_drowsy": False,
            "processed_frame": None
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    score = 0.0
    status = "AWAKE"
    is_drowsy = False

    # Draw on frame
    overlay_frame = frame.copy()
    height, width = overlay_frame.shape[:2]

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(overlay_frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Drowsiness detection logic
            if len(eyes) < 2:
                blink_counter += 1
                if blink_counter >= EAR_CONSEC_FRAMES:
                    drowsy_frames += 1
                    score = min(1.0, drowsy_frames / 10.0)
                    status = "DROWSY"
                    is_drowsy = True
            else:
                blink_counter = max(0, blink_counter - 2)
                drowsy_frames = max(0, drowsy_frames - 1)
                status = "AWAKE"
                is_drowsy = False

            fatigue_score = smooth(fatigue_score, score)
            break
    else:
        fatigue_score = smooth(fatigue_score, 0.0)
        status = "NO FACE"
        is_drowsy = False

    # Add status overlay directly on the frame
    if status == "DROWSY":
        cv2.rectangle(overlay_frame, (0, 0), (width, 70), (0, 0, 255), -1)
        cv2.putText(overlay_frame, "🚨 DROWSY - WAKE UP!", (width // 2 - 200, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    elif status == "AWAKE":
        cv2.rectangle(overlay_frame, (0, 0), (width, 70), (0, 255, 0), -1)
        cv2.putText(overlay_frame, "✅ AWAKE", (width // 2 - 80, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay_frame, (0, 0), (width, 70), (128, 128, 128), -1)
        cv2.putText(overlay_frame, "NO FACE DETECTED", (width // 2 - 160, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # Add fatigue score bottom left
    cv2.putText(overlay_frame, f"Fatigue: {fatigue_score:.2f}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Convert processed frame to base64
    _, buffer = cv2.imencode('.jpg', overlay_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "fatigue_score": float(fatigue_score),
        "status": status,
        "is_drowsy": is_drowsy,
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
        return JSONResponse({"error": str(e), "fatigue_score": 0.0, "status": "ERROR"})


@app.get("/")
async def home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    return {"status": "healthy"}