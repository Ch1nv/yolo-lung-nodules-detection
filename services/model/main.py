import base64
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from PIL import Image, ImageDraw
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "best.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DEFAULT_CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
DEFAULT_IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1280"))
RESULT_BUCKET = os.getenv("RESULT_BUCKET")
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results")

app = FastAPI(title="Lung Nodules YOLO11n Vertex Model", version="1.0.0")
storage_client: storage.Client | None = None
model: YOLO | None = None


def _storage_client() -> storage.Client:
    global storage_client
    if storage_client is None:
        storage_client = storage.Client()
    return storage_client


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket_name, object_name = uri[5:].split("/", 1)
    return bucket_name, object_name


def _read_image(instance: dict[str, Any]) -> Image.Image:
    if instance.get("image_gcs_uri"):
        bucket_name, object_name = _parse_gcs_uri(instance["image_gcs_uri"])
        image_bytes = _storage_client().bucket(bucket_name).blob(object_name).download_as_bytes()
    elif instance.get("image_base64"):
        image_bytes = base64.b64decode(instance["image_base64"])
    else:
        raise HTTPException(status_code=400, detail="Instance must include image_gcs_uri or image_base64")

    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image") from exc


def _draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=3)
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill=(255, 64, 64))
        draw.text((x1, y1), label, fill=(255, 255, 255))
    return annotated


def _upload_annotated_image(image: Image.Image) -> str | None:
    if not RESULT_BUCKET:
        return None
    object_name = f"{RESULT_PREFIX.strip('/')}/{uuid.uuid4().hex}.jpg"
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    blob = _storage_client().bucket(RESULT_BUCKET).blob(object_name)
    blob.upload_from_string(buffer.getvalue(), content_type="image/jpeg")
    return f"gs://{RESULT_BUCKET}/{object_name}"


def _encode_image_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _predict_one(instance: dict[str, Any]) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    image = _read_image(instance)
    confidence = float(instance.get("confidence", DEFAULT_CONFIDENCE))
    iou = float(instance.get("iou", DEFAULT_IOU))
    result = model.predict(
        source=np.array(image),
        imgsz=IMAGE_SIZE,
        conf=confidence,
        iou=iou,
        verbose=False,
    )[0]

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "bbox_xywh": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                }
            )

    annotated_uri = None
    annotated_base64 = None
    if instance.get("return_annotated_image", True):
        annotated = _draw_detections(image, detections)
        annotated_uri = _upload_annotated_image(annotated)
        if instance.get("return_annotated_image_base64", False):
            annotated_base64 = _encode_image_base64(annotated)

    return {
        "image_size": {"width": image.width, "height": image.height},
        "detections": detections,
        "annotated_image_gcs_uri": annotated_uri,
        "annotated_image_base64": annotated_base64,
    }


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))


@app.get("/")
@app.get("/health")
def health() -> dict[str, str | int | bool | None]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "result_bucket": RESULT_BUCKET,
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(request: Request) -> dict[str, list[dict[str, Any]]]:
    body = await request.json()
    instances = body.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="Request body must include an instances list")
    return {"predictions": [_predict_one(instance) for instance in instances]}
