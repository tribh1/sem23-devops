FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY mlops_demo ./mlops_demo
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "mlops_demo.service:app", "--host", "0.0.0.0", "--port", "8000"]
