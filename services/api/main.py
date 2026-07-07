import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import google.auth
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport.requests import Request
from google.cloud import storage
import requests


PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "asia-east1")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET")
UPLOAD_PREFIX = os.getenv("UPLOAD_PREFIX", "uploads")
SIGNED_URL_TTL_SECONDS = int(os.getenv("SIGNED_URL_TTL_SECONDS", "3600"))
ENABLE_SIGNED_URLS = os.getenv("ENABLE_SIGNED_URLS", "false").lower() == "true"
ALLOW_ORIGINS = [origin.strip() for origin in os.getenv("ALLOW_ORIGINS", "*").split(",")]

app = FastAPI(title="Lung Nodules Cloud Run API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_client: storage.Client | None = None


def _storage_client() -> storage.Client:
    global storage_client
    if storage_client is None:
        storage_client = storage.Client()
    return storage_client


def _require_settings() -> None:
    missing = []
    if not PROJECT_ID:
        missing.append("GCP_PROJECT")
    if not VERTEX_ENDPOINT_ID:
        missing.append("VERTEX_ENDPOINT_ID")
    if not UPLOAD_BUCKET:
        missing.append("UPLOAD_BUCKET")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing settings: {', '.join(missing)}")


def _safe_extension(filename: str | None) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return extension
    return ".jpg"


def _upload_image(file_bytes: bytes, filename: str | None, content_type: str | None) -> str:
    extension = _safe_extension(filename)
    object_name = f"{UPLOAD_PREFIX.strip('/')}/{uuid.uuid4().hex}{extension}"
    bucket = _storage_client().bucket(UPLOAD_BUCKET)
    blob = bucket.blob(object_name)
    blob.upload_from_string(file_bytes, content_type=content_type or "image/jpeg")
    return f"gs://{UPLOAD_BUCKET}/{object_name}"


def _credentials_token() -> str:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


def _call_vertex(instance: dict[str, Any]) -> dict[str, Any]:
    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{VERTEX_LOCATION}/endpoints/{VERTEX_ENDPOINT_ID}:predict"
    )
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {_credentials_token()}"},
        json={"instances": [instance]},
        timeout=300,
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail={"vertex_status": response.status_code, "body": response.text})
    payload = response.json()
    predictions = payload.get("predictions") or []
    if not predictions:
        raise HTTPException(status_code=502, detail="Vertex AI returned no predictions")
    return predictions[0]


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket_name, object_name = uri[5:].split("/", 1)
    return bucket_name, object_name


def _try_signed_url(uri: str) -> str | None:
    if not ENABLE_SIGNED_URLS:
        return None
    try:
        bucket_name, object_name = _parse_gcs_uri(uri)
        blob = _storage_client().bucket(bucket_name).blob(object_name)
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=SIGNED_URL_TTL_SECONDS),
            method="GET",
        )
    except Exception:
        return None


@app.get("/health")
def health() -> dict[str, str | bool | None]:
    return {
        "status": "ok",
        "project_id": PROJECT_ID,
        "vertex_location": VERTEX_LOCATION,
        "vertex_endpoint_id": VERTEX_ENDPOINT_ID,
        "upload_bucket": UPLOAD_BUCKET,
        "signed_urls_enabled": ENABLE_SIGNED_URLS,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    return_annotated_image: bool = Form(True),
    return_annotated_image_base64: bool = Form(False),
) -> dict[str, Any]:
    _require_settings()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image_gcs_uri = _upload_image(image_bytes, file.filename, file.content_type)
    prediction = _call_vertex(
        {
            "image_gcs_uri": image_gcs_uri,
            "confidence": confidence,
            "iou": iou,
            "return_annotated_image": return_annotated_image,
            "return_annotated_image_base64": return_annotated_image_base64,
        }
    )

    annotated_uri = prediction.get("annotated_image_gcs_uri")
    return {
        "filename": file.filename,
        "image_gcs_uri": image_gcs_uri,
        "detections": prediction.get("detections", []),
        "image_size": prediction.get("image_size"),
        "annotated_image_gcs_uri": annotated_uri,
        "annotated_image_url": _try_signed_url(annotated_uri) if annotated_uri else None,
        "annotated_image_base64": prediction.get("annotated_image_base64"),
    }
