#!/bin/bash
# Know-Defeat: Google Cloud Run Setup Script
# This script sets up the necessary GCP resources for deploying to Cloud Run

# Exit on error
set -e

# Check if variables are provided
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ]; then
  echo "Usage: PROJECT_ID=your-project-id REGION=your-region ./setup_cloud_run.sh"
  echo "Example: PROJECT_ID=my-project-123 REGION=us-central1 ./setup_cloud_run.sh"
  exit 1
fi

# Set APP_NAME
APP_NAME=know-defeat

echo "Setting up Cloud Run deployment for $APP_NAME in project $PROJECT_ID, region $REGION"

# Enable required Google Cloud APIs
echo "Enabling required APIs..."
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable sqladmin.googleapis.com

# Create a service account for GitHub Actions (if it doesn't exist)
echo "Setting up service account for GitHub Actions..."
if gcloud iam service-accounts describe github-actions@$PROJECT_ID.iam.gserviceaccount.com > /dev/null 2>&1; then
  echo "Service account github-actions already exists, skipping creation."
else
  echo "Creating service account github-actions..."
  gcloud iam service-accounts create github-actions
fi

# Assign necessary permissions
echo "Assigning IAM permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# For Cloud SQL access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Create a key file for GitHub Actions
echo "Creating credentials file..."
gcloud iam service-accounts keys create credentials.json \
  --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com

echo ""
echo "========================================================"
echo "Setup complete! Next steps:"
echo "========================================================"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - GCP_PROJECT_ID: $PROJECT_ID"
echo "   - GCP_CREDENTIALS: (content of credentials.json)"
echo "   - GCP_REGION: $REGION"
echo ""
echo "2. For database connectivity, also add:"
echo "   - DB_PASSWORD: (your database password)"
echo "   - CLOUD_SQL_CONNECTION_NAME: (your SQL instance connection name)"
echo ""
echo "3. Push your code to GitHub to trigger the deployment"
echo ""
echo "The credentials.json file has been created in the current directory."
echo "IMPORTANT: Keep this file secure and delete it after adding to GitHub secrets!"
echo "========================================================="