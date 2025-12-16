FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_service.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "ocr_service.py"]
