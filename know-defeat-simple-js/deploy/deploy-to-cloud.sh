#!/bin/bash
# Script to deploy Know Defeat Trading System to Google Cloud

# Set project variables
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
DB_INSTANCE="know-defeat-db"
API_SERVICE="know-defeat-api"
UI_SERVICE="know-defeat-ui"

# Ensure the Google Cloud SDK is installed
if ! command -v gcloud &> /dev/null
then
    echo "Google Cloud SDK not found. Please install it first."
    exit 1
fi

# Set the current project
echo "Setting current project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com sqladmin.googleapis.com

# Deploy the API
echo "Deploying API service to Cloud Run..."
cd ../api
gcloud run deploy $API_SERVICE \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "DB_HOST=/cloudsql/$PROJECT_ID:$REGION:$DB_INSTANCE,DB_NAME=tick_data,NODE_ENV=production"

# Get the API URL for the UI deployment
API_URL=$(gcloud run services describe $API_SERVICE --platform managed --region $REGION --format 'value(status.url)')

# Deploy the UI
echo "Deploying UI service to Cloud Run..."
cd ../ui
gcloud run deploy $UI_SERVICE \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "VITE_API_URL=$API_URL/api"

# Get the UI URL
UI_URL=$(gcloud run services describe $UI_SERVICE --platform managed --region $REGION --format 'value(status.url)')

echo "Deployment complete!"
echo "API URL: $API_URL"
echo "UI URL: $UI_URL"