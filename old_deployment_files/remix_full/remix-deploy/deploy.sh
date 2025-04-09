#!/bin/bash

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-ui"

# Set project ID
gcloud config set project ${PROJECT_ID}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --source . \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars "NODE_ENV=production,USE_MOCK_DATA=true" \
  --max-instances=2 \
  --memory=512Mi \
  --timeout=30s
