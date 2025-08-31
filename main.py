import logging
import threading
import atexit
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

# -------------------------
# ✅ Workaround for Python 3.13 threading bug
# -------------------------
def cleanup_threads():
    for t in threading.enumerate():
        if t.is_alive() and t.daemon:
            try:
                t.join(timeout=0.1)
            except Exception:
                pass

atexit.register(cleanup_threads)

# -------------------------
# ✅ Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------
# ✅ Load YOLO model (once at startup)
# -------------------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # dict: {0: "acne", 1: "eczema", ...}

# -------------------------
# ✅ FastAPI setup
# -------------------------
app = FastAPI(title="YOLO Inference API")

# Define allowed origins for CORS
ALLOWED_ORIGINS = [
   
    "http://localhost:5173",  # Another possible dev server port (e.g., Vite)
    "https://niraadevice.netlify.app",  # Production frontend domain
   
    # Add more domains as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Explicitly list allowed domains
    allow_credentials=True,  # Allow cookies/credentials if needed
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow methods used
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "X-Requested-With",
    ],  # Allow specific headers
)

# -------------------------
# ✅ API Route - Upload & Detect
# -------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        results = model.predict(image, conf=0.25, iou=0.45)

        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, cls_ids):
                detections.append({
                    "class_id": int(cls),
                    "class_name": CLASS_NAMES[int(cls)],
                    "confidence": float(conf),
                    "box": [float(x) for x in box]
                })

        logging.info(f"✅ Processed: {file.filename}, Detections: {len(detections)}")

        return {
            "filename": file.filename,
            "detections": detections
        }

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return {"error": str(e)}

# -------------------------
# ✅ Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
