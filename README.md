# 🐱🐶 Dog Cat Classifier

Backend FastAPI dùng để phân loại ảnh chó và mèo bằng mô hình CNN từ TensorFlow và Keras đã huấn luyện.

## Tính năng

- Upload ảnh để phân loại
- Dự đoán chó hoặc mèo theo thời gian thực
- API RESTful với tài liệu tự động
- Hỗ trợ CORS để kết nối với frontend
- Endpoint kiểm tra trạng thái hệ thống

## Hướng dẫn cài đặt

1. Tạo môi trường .venv và cài đặt các thư viện Python:
```bash
python -m venv .venv
pip install -r requirements.txt
```

2. Đảm bảo file mô hình `dog_cat_classifier_final_no_opt.h5` nằm trong thư mục backend.

3. Khởi động máy chủ FastAPI:
```bash
.venv\Scripts\activate
python main.py
```

Hoặc chạy trực tiếp bằng uvicorn:
```bash
.venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Các endpoint API

### GET /
Kiểm tra trạng thái hệ thống

### GET /model/info
Lấy thông tin về mô hình đã tải

### POST /predict
Upload ảnh để phân loại

**Yêu cầu:**
- Phương thức: POST
- Content-Type: multipart/form-data
- Body: file (ảnh)

**Phản hồi:**
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

## Thông tin mô hình

- Đầu vào: Ảnh RGB kích thước 224x224x3
- Đầu ra: Giá trị từ 0 đến 1
  - > 0.5: Chó
  - <= 0.5: Mèo
- Định dạng mô hình: File TensorFlow/Keras (.h5)

## Tài liệu API

Khi server đang chạy, truy cập:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc