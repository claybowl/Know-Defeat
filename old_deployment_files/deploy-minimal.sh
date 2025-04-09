#!/bin/bash
# Ultra minimal deployment for Know Defeat Frontend

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-minimal"

# Set project ID
gcloud config set project ${PROJECT_ID}

# Build and deploy using Cloud Build
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME} --dockerfile Dockerfile.minimal

# Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 512Mi \
  --port 8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo "Deployment complete! Your application is available at: ${SERVICE_URL}"
echo "Check logs with: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"