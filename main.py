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

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Dog Cat Classifier API", version="1.0.0")

# Thêm middleware CORS để cho phép frontend truy cập API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Khi triển khai thực tế, nên chỉ định URL frontend cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình mô hình
IMG_WIDTH = 224
IMG_HEIGHT = 224
MODEL_PATH = "dog_cat_classifier_final_no_opt.h5"

# Biến model toàn cục
model = None

@app.on_event("startup")
async def load_model():
    """Load mô hình đã huấn luyện khi khởi động server"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"Đã load mô hình thành công từ {MODEL_PATH}")
        else:
            print(f"Không tìm thấy file mô hình tại {MODEL_PATH}")
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi khi load mô hình: {e}")
        raise e

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Tiền xử lý ảnh được upload để chuẩn bị cho việc dự đoán
    
    Tham số:
        image_bytes: Dữ liệu ảnh thô dạng bytes
        
    Trả về:
        Mảng ảnh đã được tiền xử lý sẵn sàng cho dự đoán
    """
    try:
        # Chuyển đổi bytes thành ảnh PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Chuyển sang RGB nếu cần thiết
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Chuyển ảnh PIL thành mảng numpy
        img_array = np.array(image)
        
        # Resize ảnh về kích thước chuẩn
        img_resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        
        # Chuẩn hóa giá trị pixel về khoảng [0, 1]
        img_normalized = img_resized / 255.0
        
        # Thêm chiều batch
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh: {str(e)}")

def predict_image(img_array: np.ndarray) -> Dict[str, float]:
    """
    Dự đoán trên ảnh đã được tiền xử lý
    
    Tham số:
        img_array: Mảng ảnh đã tiền xử lý
        
    Trả về:
        Dictionary chứa kết quả dự đoán
    """
    try:
        # Thực hiện dự đoán
        prediction = model.predict(img_array)[0][0]
        
        # Chuyển đổi sang tên lớp và độ tin cậy
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
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")

@app.get("/")
async def root():
    """Endpoint kiểm tra trạng thái hệ thống"""
    return {"message": "Dog Cat Classifier API đang chạy", "status": "healthy"}

@app.get("/model/info")
async def get_model_info():
    """Lấy thông tin về mô hình đã load"""
    if model is None:
        raise HTTPException(status_code=503, detail="Chưa load mô hình")
    
    return {
        "model_path": MODEL_PATH,
        "input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3],
        "classes": ["cat", "dog"],
        "loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Dự đoán ảnh upload là chó hay mèo
    
    Tham số:
        file: File ảnh được upload
        
    Trả về:
        JSON chứa kết quả dự đoán
    """
    # Kiểm tra mô hình đã load chưa
    if model is None:
        raise HTTPException(status_code=503, detail="Chưa load mô hình. Vui lòng khởi động lại server.")
    
    # Kiểm tra loại file upload có phải là ảnh không
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phải là ảnh")
    
    try:
        # Đọc dữ liệu ảnh từ file upload
        image_bytes = await file.read()
        
        # Tiền xử lý ảnh
        img_array = preprocess_image(image_bytes)
        
        # Dự đoán kết quả
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
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ server: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)