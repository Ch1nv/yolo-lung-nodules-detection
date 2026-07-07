# Deploy YOLO11n Lung Nodules API to GCP

This project is set up for the two-service architecture:

```text
Cloud Run API -> Cloud Storage -> Vertex AI GPU endpoint -> Cloud Storage results
```

The trained model is `results/yolo11n/weights/best.pt`.

## 1. Local smoke test

The root `Dockerfile` builds the Vertex AI model container. The API container uses `docker/api.Dockerfile`.

Model container:

```powershell
docker build -t lung-nodules-yolo11n-model .
docker run --rm -p 8080:8080 lung-nodules-yolo11n-model
```

API container:

```powershell
docker build -f docker/api.Dockerfile -t lung-nodules-api .
```

Local model request with base64 is useful before connecting GCS:

```powershell
$IMAGE_PATH="ct_images/images/val/1_jpg.rf.4a59a63d0a7339d280dd18ef3c2e675a.jpg"
$B64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($IMAGE_PATH))
$BODY=@{ instances=@(@{ image_base64=$B64; confidence=0.25; iou=0.45; return_annotated_image=$true; return_annotated_image_base64=$true }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri "http://localhost:8080/predict" -ContentType "application/json" -Body $BODY
```

## 2. Choose GCP settings

Suggested settings:

```powershell
$PROJECT_ID="your-gcp-project-id"
$REGION="asia-east1"
$REPOSITORY="lung-nodules"
$API_SERVICE_NAME="lung-nodules-api"
$MODEL_IMAGE_NAME="lung-nodules-yolo11n-model"
$MODEL_DISPLAY_NAME="lung-nodules-yolo11n"
$UPLOAD_BUCKET="$PROJECT_ID-lung-nodules-uploads"
$RESULT_BUCKET="$PROJECT_ID-lung-nodules-results"
gcloud config set project $PROJECT_ID
```

`asia-east1` is Taiwan. Use another region if your users or compliance requirements point elsewhere.

## 3. Enable APIs

```powershell
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  aiplatform.googleapis.com `
  iamcredentials.googleapis.com `
  storage.googleapis.com
```

## 4. Create buckets and Artifact Registry

Run once:

```powershell
gcloud artifacts repositories create $REPOSITORY `
  --repository-format=docker `
  --location=$REGION `
  --description="Docker images for lung nodules detection"

gcloud storage buckets create "gs://$UPLOAD_BUCKET" --location=$REGION --uniform-bucket-level-access
gcloud storage buckets create "gs://$RESULT_BUCKET" --location=$REGION --uniform-bucket-level-access
```

## 5. Create service accounts

```powershell
gcloud iam service-accounts create lung-nodules-api
gcloud iam service-accounts create lung-nodules-model

$API_SA="lung-nodules-api@$PROJECT_ID.iam.gserviceaccount.com"
$MODEL_SA="lung-nodules-model@$PROJECT_ID.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://$UPLOAD_BUCKET" `
  --member="serviceAccount:$API_SA" `
  --role="roles/storage.objectCreator"
gcloud storage buckets add-iam-policy-binding "gs://$UPLOAD_BUCKET" `
  --member="serviceAccount:$MODEL_SA" `
  --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding "gs://$RESULT_BUCKET" `
  --member="serviceAccount:$MODEL_SA" `
  --role="roles/storage.objectCreator"
gcloud storage buckets add-iam-policy-binding "gs://$RESULT_BUCKET" `
  --member="serviceAccount:$API_SA" `
  --role="roles/storage.objectViewer"
```

Grant the API service account permission to call the Vertex endpoint after the endpoint is created:

```powershell
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$API_SA" `
  --role="roles/aiplatform.user"
```

## 6. Build and push images manually

```powershell
$API_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$API_SERVICE_NAME:latest"
$MODEL_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$MODEL_IMAGE_NAME:latest"

gcloud auth configure-docker "$REGION-docker.pkg.dev"
docker build -f docker/api.Dockerfile -t $API_IMAGE .
docker push $API_IMAGE

docker build -t $MODEL_IMAGE .
docker push $MODEL_IMAGE
```

## 7. Create and deploy the Vertex AI endpoint

Create the endpoint once:

```powershell
gcloud ai endpoints create `
  --region=$REGION `
  --display-name="lung-nodules-yolo11n-endpoint"
```

Copy the endpoint ID printed by the command:

```powershell
$VERTEX_ENDPOINT_ID="your-endpoint-id"
```

Upload and deploy the model:

```powershell
gcloud ai models upload `
  --region=$REGION `
  --display-name=$MODEL_DISPLAY_NAME `
  --container-image-uri=$MODEL_IMAGE `
  --container-predict-route="/predict" `
  --container-health-route="/health" `
  --container-ports=8080 `
  --container-env-vars="RESULT_BUCKET=$RESULT_BUCKET"

$MODEL_NAME=(gcloud ai models list `
  --region=$REGION `
  --filter="displayName=$MODEL_DISPLAY_NAME" `
  --sort-by="~createTime" `
  --limit=1 `
  --format="value(name)")
$MODEL_ID=$MODEL_NAME.Split("/")[-1]

gcloud ai endpoints deploy-model $VERTEX_ENDPOINT_ID `
  --region=$REGION `
  --model=$MODEL_ID `
  --display-name=$MODEL_DISPLAY_NAME `
  --machine-type="n1-standard-4" `
  --accelerator="type=nvidia-tesla-t4,count=1" `
  --min-replica-count=1 `
  --max-replica-count=1 `
  --service-account=$MODEL_SA `
  --traffic-split=0=100
```

T4 is a reasonable first GPU for YOLO11n. You can adjust machine type and accelerator based on quota and latency.

## 8. Deploy Cloud Run API

```powershell
gcloud run deploy $API_SERVICE_NAME `
  --image=$API_IMAGE `
  --region=$REGION `
  --platform=managed `
  --service-account=$API_SA `
  --memory=1Gi `
  --cpu=1 `
  --timeout=300 `
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,VERTEX_LOCATION=$REGION,VERTEX_ENDPOINT_ID=$VERTEX_ENDPOINT_ID,UPLOAD_BUCKET=$UPLOAD_BUCKET,RESULT_BUCKET=$RESULT_BUCKET" `
  --allow-unauthenticated
```

Remove `--allow-unauthenticated` before using real private medical data.

## 9. Test the deployed API

```powershell
$URL="https://your-cloud-run-url"
curl "$URL/health"
curl -X POST "$URL/predict" -F "file=@ct_images/images/val/1_jpg.rf.4a59a63d0a7339d280dd18ef3c2e675a.jpg"
```

For a quick frontend demo without signed URLs, request a base64 annotated image:

```powershell
curl -X POST "$URL/predict" `
  -F "file=@ct_images/images/val/1_jpg.rf.4a59a63d0a7339d280dd18ef3c2e675a.jpg" `
  -F "return_annotated_image_base64=true"
```

## 10. GitHub Actions setup

The workflows in `.github/workflows` expect GitHub Actions variables:

```text
GCP_PROJECT_ID
GCP_REGION
ARTIFACT_REPOSITORY
GCP_WORKLOAD_IDENTITY_PROVIDER
GCP_DEPLOY_SERVICE_ACCOUNT
CLOUD_RUN_SERVICE
CLOUD_RUN_SERVICE_ACCOUNT
UPLOAD_BUCKET
RESULT_BUCKET
VERTEX_LOCATION
VERTEX_ENDPOINT_ID
VERTEX_MODEL_IMAGE_NAME
VERTEX_MODEL_DISPLAY_NAME
VERTEX_MODEL_SERVICE_ACCOUNT
VERTEX_MACHINE_TYPE
VERTEX_ACCELERATOR_TYPE
VERTEX_ACCELERATOR_COUNT
```

Recommended initial values:

```text
GCP_REGION=asia-east1
CLOUD_RUN_SERVICE=lung-nodules-api
VERTEX_LOCATION=asia-east1
VERTEX_MODEL_IMAGE_NAME=lung-nodules-yolo11n-model
VERTEX_MODEL_DISPLAY_NAME=lung-nodules-yolo11n
VERTEX_MACHINE_TYPE=n1-standard-4
VERTEX_ACCELERATOR_TYPE=nvidia-tesla-t4
VERTEX_ACCELERATOR_COUNT=1
```

Use Workload Identity Federation for GitHub authentication instead of a JSON service account key.
