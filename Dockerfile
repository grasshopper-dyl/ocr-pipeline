FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime



WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_service.py .

# (optional but nice)
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "ocr_service.py"]