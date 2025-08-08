# Use a slim Python with pip; CPU is fine, switch to CUDA base if you want GPU
FROM python:3.11-slim

# System deps for OCR & PDF rendering
# - tesseract-ocr + Russian language pack
# - poppler-utils gives pdftoppm used by pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .

# Spaces will run: python app.py
CMD ["python", "app.py"]
