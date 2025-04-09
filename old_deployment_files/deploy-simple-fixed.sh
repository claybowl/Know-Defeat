#!/bin/bash
# Super simple deployment script for Know Defeat Frontend to Google Cloud Run

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-frontend-minimal"  # Using a new service name to avoid conflicts

# Ensure Google Cloud SDK is authenticated
echo "Checking Google Cloud authentication..."
gcloud auth list

# Set project ID
echo "Setting project to: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# Submit docker build using the simple Dockerfile
echo "Building Docker image with simplified Dockerfile..."
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME} --dockerfile Dockerfile.simple

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 512Mi \
  --port 8080 \
  --set-env-vars "NODE_ENV=production,USE_MOCK_DATA=true"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Deployment complete!"
echo "Your application is available at: ${SERVICE_URL}"

# Provide testing instructions
echo ""
echo "===================================="
echo "IMPORTANT: Debugging Notes"
echo "===================================="
echo "1. This is a minimalistic deployment with mock data"
echo "2. Check logs with: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"
echo "3. Start with testing: ${SERVICE_URL}/healthcheck to verify the service is running"