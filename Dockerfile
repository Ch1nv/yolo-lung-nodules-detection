FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODEL_PATH=/app/model/best.pt \
    IMAGE_SIZE=1280

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-model.txt .
RUN pip install --no-cache-dir -r requirements-model.txt

RUN mkdir -p /app/model /tmp/Ultralytics

COPY services/model/main.py .
COPY results/yolo11n/weights/best.pt /app/model/best.pt

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /tmp/Ultralytics

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
