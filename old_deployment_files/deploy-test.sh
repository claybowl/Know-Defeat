#!/bin/bash
# Test deployment to verify Google Cloud setup

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-test"

# Set project ID
gcloud config set project ${PROJECT_ID}

# Path to this directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Working from directory: $DIR"

# Build test image explicitly (using --no-source)
echo "Building test image..."
gcloud builds submit --no-source --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --config - <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/${PROJECT_ID}/${SERVICE_NAME}', '-f', 'Dockerfile.test', '.']
  dir: '/workspace'
timeout: '1200s'
EOF

# Check if the image was created
echo "Checking if image was created..."
gcloud container images list-tags gcr.io/${PROJECT_ID}/${SERVICE_NAME} --limit=1

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 512Mi \
  --port 8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo "Deployment complete! Your test server is available at: ${SERVICE_URL}"
echo "Check logs with: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"