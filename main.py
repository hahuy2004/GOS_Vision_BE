from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import os
from typing import Dict

# Initialize FastAPI app
app = FastAPI(title="Dog Cat Classifier API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
IMG_WIDTH = 150
IMG_HEIGHT = 150
MODEL_PATH = "dog_cat_classifier_final.keras"

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess the uploaded image for prediction
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Resize image
        img_resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_image(img_array: np.ndarray) -> Dict[str, float]:
    """
    Make prediction on preprocessed image
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Convert to class name and confidence
        if prediction > 0.5:
            class_name = 'dog'
            confidence = float(prediction)
        else:
            class_name = 'cat'
            confidence = float(1 - prediction)
        
        return {
            "class": class_name,
            "confidence": confidence,
            "raw_prediction": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Dog Cat Classifier API is running", "status": "healthy"}

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3],
        "classes": ["cat", "dog"],
        "loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether the uploaded image is a cat or dog
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please restart the server.")
    
    # Check file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_image(img_array)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": file.filename,
                "prediction": result
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)