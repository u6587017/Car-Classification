import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model

app = FastAPI(title="Car Type Classifier API")

# --- GLOBAL MODEL LOADING ---
# Load once when the script starts, not inside the function.
MODEL_PATH = os.path.join("./models", "resnet50v2_model.keras")

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # In production, you'd want to handle this more gracefully
    model = None

def preprocess_image(image_bytes):
    # Convert bytes to a numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image.")

    # Processing logic
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(img_rgb, (256, 256))
    normalized = resized / 255.0
    return np.expand_dims(normalized, 0)

@app.post("/classify")
async def predict_car_type(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # 2. Read image bytes
        image_bytes = await file.read()
        
        # 3. Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # 4. Predict
        predictions = model.predict(input_tensor)
        
        # 5. Format Output
        # Assuming your model returns a list of probabilities
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "alive", "model_loaded": model is not None}