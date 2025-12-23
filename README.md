# Demo MLOps pipeline

Dự án minh họa một luồng MLOps tối giản cho bài toán phân loại Iris:

1. **Huấn luyện mô hình**: Pipeline sử dụng `scikit-learn` để huấn luyện logistic regression và lưu mô hình, metric.
2. **CI/CD**: Workflow GitHub Actions kiểm tra style cơ bản và chạy test trước khi deploy.
3. **Triển khai & suy luận**: Dịch vụ FastAPI tải mô hình đã lưu và cung cấp API dự đoán.
4. **Theo dõi/giám sát**: Mọi request được ghi lại để tính thống kê đơn giản (độ tự tin trung bình, phân phối lớp, drift score).

## Chuẩn bị môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Huấn luyện mô hình

```bash
python -m mlops_demo.pipeline
```

Artifacts được lưu trong thư mục `artifacts/` gồm mô hình (`model.joblib`), metric (`metrics.json`) và cấu hình chạy (`run_config.json`).

## Chạy dịch vụ suy luận

```bash
uvicorn mlops_demo.service:app --reload --host 0.0.0.0 --port 8000
```

Các endpoint chính:

- `GET /health` – kiểm tra tình trạng dịch vụ.
- `POST /predict` – nhận 4 đặc trưng Iris và trả về lớp dự đoán kèm độ tự tin.
- `GET /metrics` – thống kê monitoring nội bộ (số lượng request, phân phối lớp, drift score).

## Kiểm thử

```bash
pytest
```

## CI/CD

Workflow tại `.github/workflows/ci.yml` sẽ cài đặt dependency và chạy `pytest` cho mỗi commit/pull request. Đây là bước kiểm tra tự động trước khi deploy.

Workflow CD (`.github/workflows/cd.yml`) chạy trên nhánh `main` hoặc khi `workflow_dispatch`, thực hiện:

- Cài đặt dependency và chạy toàn bộ test.
- Build Docker image cho dịch vụ FastAPI.
- Đóng gói image thành artifact (`iris-mlops-image`) sẵn sàng cho bước deploy tiếp theo (push registry/triển khai môi trường chạy container).
