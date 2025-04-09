#!/bin/bash
# Script to deploy just the UI component

# Configuration
PROJECT_ID="know-defeat-trading-js"
REGION="us-central1"
UI_SERVICE="know-defeat-ui"
API_URL="https://know-defeat-api-fqag4rwuia-uc.a.run.app"

# Ensure we're using the right project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Update the UI configuration with the API URL
echo "Updating UI environment to use API at $API_URL..."
cd ../ui
# Update the API URL in cloudbuild.yaml
sed -i "s|VITE_API_URL=.*|VITE_API_URL=${API_URL}/api'|g" cloudbuild.yaml

# Deploy UI service
echo "Deploying UI service..."
gcloud builds submit --config=cloudbuild.yaml

# Get the UI service URL
echo "Getting UI service URL..."
UI_URL=$(gcloud run services describe $UI_SERVICE --platform managed --region $REGION --format 'value(status.url)')

echo "==============================================="
echo "UI Deployment Complete!"
echo "UI URL: $UI_URL"
echo "==============================================="