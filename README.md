# Dog Cat Classifier Backend

FastAPI backend for classifying dog and cat images using a pre-trained TensorFlow model.

## Features

- Upload image files for classification
- Real-time prediction of dog vs cat
- RESTful API with automatic documentation
- CORS enabled for frontend integration
- Health check endpoints

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model file `dog_cat_classifier_final.h5` is in the backend directory.

3. Run the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### GET /
Health check endpoint

### GET /model/info
Get information about the loaded model

### POST /predict
Upload an image file for classification

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
  "success": true,
  "filename": "example.jpg",
  "prediction": {
    "class": "dog",
    "confidence": 0.85,
    "raw_prediction": 0.85
  }
}
```

## Model Details

- Input shape: 150x150x3 (RGB images)
- Output: Single value between 0 and 1
  - > 0.5: Dog
  - <= 0.5: Cat
- Model format: TensorFlow/Keras .h5 file

## API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc