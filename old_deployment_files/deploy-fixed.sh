#!/bin/bash
# Streamlined deployment script for Know Defeat Frontend to Google Cloud Run

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-frontend-fixed"  # Using a new service name to avoid conflicts

# Ensure Google Cloud SDK is authenticated
echo "Checking Google Cloud authentication..."
gcloud auth list

# Set project ID
echo "Setting project to: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# Clean build if exists
echo "Cleaning previous build..."
rm -rf build

# Force rebuild with CommonJS format
echo "Ensuring vite.config.js is using CommonJS format..."
sed -i 's/serverModuleFormat: .esm./serverModuleFormat: "cjs"/' vite.config.js

# Build Docker image with complete build
echo "Building and deploying with Cloud Build and Run..."
gcloud run deploy ${SERVICE_NAME} \
  --source . \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 512Mi \
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
echo "1. This is a simplified deployment with only mock data"
echo "2. Check logs with: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"
echo "3. To view the logs in realtime: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --follow"
echo "4. Visit ${SERVICE_URL}/healthcheck to verify the service is running"