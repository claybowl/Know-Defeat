#!/bin/bash
# Simple deployment script for Know Defeat Frontend to Google Cloud Run

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-frontend"

# Ensure Google Cloud SDK is authenticated
echo "Checking Google Cloud authentication..."
gcloud auth list

# Set project ID
echo "Setting project to: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# Build Docker image
echo "Building Docker image..."
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 512Mi \
  --set-env-vars "USE_MOCK_DATA=true"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Deployment complete!"
echo "Your application is available at: ${SERVICE_URL}"