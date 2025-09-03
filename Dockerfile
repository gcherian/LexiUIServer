FROM python:3.11-slim

# System deps for OCR + PDF rasterization
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng poppler-utils libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api /app/api

# OpenShift-friendly perms
RUN mkdir -p /app/data /app/out && chgrp -R 0 /app && chmod -R g+rwX /app

EXPOSE 8000
USER 1001

CMD ["python","-m","uvicorn","api.api:app","--host","0.0.0.0","--port","8000"]
