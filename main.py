from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import cv2
import numpy as np
from typing import List, Dict

app = FastAPI(
    title="Complete Face Detection API",
    description="Backend API to detect faces from an uploaded image. Includes endpoints for both data (coordinates) and graphical (image) responses.",
    version="1.1.0"
)

# Configure CORS to allow any frontend (React, Vue, HTML/JS) to communicate with this backend without browser blocking!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins in development. This is critical for frontends on different ports.
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, OPTIONS)
    allow_headers=["*"], # Allow all custom headers
)

# Load the pre-trained OpenCV Haar cascade model for frontal face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.get("/", summary="Health Check")
async def health_check():
    """Returns a simple status message indicating the API is running."""
    return {"status": "success", "message": "Face Detection API is up and running!"}

@app.post("/api/detect-faces", summary="Get Face Coordinates")
async def detect_faces(file: UploadFile = File(...)) -> List[Dict[str, int]]:
    """
    Accepts a `.jpg/.png` file object upload via `multipart/form-data`.
    Returns a JSON list of bounded coordinates for each detected face.
    """
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")
    
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    for (x, y, w, h) in faces:
        results.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })
        
    return results

@app.post("/api/detect-image", summary="Get Processed Face Image")
async def detect_image(file: UploadFile = File(...)):
    """
    Accepts a `.jpg/.png` file object upload.
    Detects faces, draws bright green rectangles around them, and returns the modified image binary file back to the client.
    Very useful when frontends simply want to flash the finished result without managing drawing coordinates manually.
    """
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")
        
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw green rectangles directly on the image
    for (x, y, w, h) in faces:
        # We specify (x,y) for origin and (x+w, y+h) for opposite corner. Color is (B, G, R) so passing 255 for Green
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Re-encode image to JPEG from numpy array
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode the processed image.")
        
    # Return directly as an image/jpeg HTTP response 
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
