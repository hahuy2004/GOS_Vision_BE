from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import os
from typing import Dict

# Biến model toàn cục
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng - khởi tạo và dọn dẹp tài nguyên"""
    # Khởi tạo - Load model khi ứng dụng bắt đầu
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model đã được load thành công từ {MODEL_PATH}")
        else:
            print(f"Không tìm thấy file model tại {MODEL_PATH}")
            raise FileNotFoundError(f"Không tìm thấy file model tại {MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        raise e
    
    yield  # Ứng dụng chạy ở đây
    
    # Dọn dẹp - Có thể thêm code dọn dẹp tài nguyên ở đây nếu cần
    print("Đang đóng ứng dụng...")

# Khởi tạo ứng dụng FastAPI với lifespan event handler
app = FastAPI(
    title="Dog Cat Classifier API", 
    version="1.0.0",
    lifespan=lifespan
)

# Thêm CORS middleware để cho phép truy cập từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, nên chỉ định URL frontend cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình model
IMG_WIDTH = 224
IMG_HEIGHT = 224
MODEL_PATH = "dog_cat_classifier_final_no_opt.h5"

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Tiền xử lý ảnh được upload để chuẩn bị cho việc dự đoán
    
    Args:
        image_bytes: Dữ liệu ảnh thô dạng bytes
        
    Returns:
        Mảng ảnh đã được tiền xử lý sẵn sàng cho dự đoán
    """
    try:
        # Chuyển đổi bytes thành PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Chuyển đổi sang RGB nếu cần thiết
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Chuyển đổi PIL thành numpy array
        img_array = np.array(image)
        
        # Resize ảnh về kích thước chuẩn
        img_resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        
        # Chuẩn hóa giá trị pixel về khoảng [0, 1]
        img_normalized = img_resized / 255.0
        
        # Thêm dimension batch
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi xử lý ảnh: {str(e)}")

def predict_image(img_array: np.ndarray) -> Dict[str, float]:
    """
    Thực hiện dự đoán trên ảnh đã được tiền xử lý
    
    Args:
        img_array: Mảng ảnh đã được tiền xử lý
        
    Returns:
        Dictionary chứa kết quả dự đoán
    """
    try:
        # Thực hiện dự đoán
        prediction = model.predict(img_array)[0][0]
        
        # Chuyển đổi thành tên class và độ tin cậy
        if prediction > 0.5:
            class_name = 'dog'  # chó
            confidence = float(prediction)
        else:
            class_name = 'cat'  # mèo
            confidence = float(1 - prediction)
        
        return {
            "class": class_name,
            "confidence": confidence,
            "raw_prediction": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình dự đoán: {str(e)}")

@app.get("/")
async def root():
    """Endpoint kiểm tra trạng thái sức khỏe của API"""
    return {"message": "Dog Cat Classifier API đang hoạt động", "status": "healthy"}

@app.get("/model/info")
async def get_model_info():
    """Lấy thông tin về model đã được load"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    
    return {
        "model_path": MODEL_PATH,
        "input_shape": [None, IMG_HEIGHT, IMG_WIDTH, 3],
        "classes": ["cat", "dog"],  # ["mèo", "chó"]
        "loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Dự đoán ảnh được upload là mèo hay chó
    
    Args:
        file: File ảnh được upload
        
    Returns:
        JSON response chứa kết quả dự đoán
    """
    # Kiểm tra xem model đã được load chưa
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load. Vui lòng khởi động lại server.")
    
    # Kiểm tra loại file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phải là một ảnh")
    
    try:
        # Đọc dữ liệu ảnh
        image_bytes = await file.read()
        
        # Tiền xử lý ảnh
        img_array = preprocess_image(image_bytes)
        
        # Thực hiện dự đoán
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
        raise HTTPException(status_code=500, detail=f"Lỗi server nội bộ: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)