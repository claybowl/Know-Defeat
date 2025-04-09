#!/bin/bash
# Test app deployment

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-test-app"

# Set project ID
gcloud config set project ${PROJECT_ID}

# Build and deploy using Cloud Run directly (simplest approach)
gcloud run deploy ${SERVICE_NAME} \
  --source . \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo "Deployment complete! Your test app is available at: ${SERVICE_URL}"