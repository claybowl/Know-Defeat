#!/bin/bash
# Deployment script for Know Defeat Trading System

# Configuration
PROJECT_ID="know-defeat-trading-js"
REGION="us-central1"
API_SERVICE="know-defeat-api"
UI_SERVICE="know-defeat-ui"

# Ensure we're using the right project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required services if not already enabled
echo "Enabling required GCP services..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com

# Deploy API service
echo "Deploying API service..."
cd ../api
gcloud builds submit --config=cloudbuild.yaml

# Get the API service URL
echo "Getting API service URL..."
API_URL=$(gcloud run services describe $API_SERVICE --platform managed --region $REGION --format 'value(status.url)')
echo "API deployed at: $API_URL"

# Update the UI configuration with the API URL
echo "Updating UI environment to use API at $API_URL..."
cd ../ui
# Update env variable in cloudbuild.yaml
sed -i "s|VITE_API_URL=.*|VITE_API_URL=${API_URL}/api|g" cloudbuild.yaml

# Deploy UI service
echo "Deploying UI service..."
gcloud builds submit --config=cloudbuild.yaml

# Get the UI service URL
echo "Getting UI service URL..."
UI_URL=$(gcloud run services describe $UI_SERVICE --platform managed --region $REGION --format 'value(status.url)')

echo "==============================================="
echo "Deployment Complete!"
echo "API URL: $API_URL"
echo "UI URL: $UI_URL"
echo "==============================================="