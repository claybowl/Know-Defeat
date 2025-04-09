#!/bin/bash
# Use Google Cloud Build to handle the entire build and deployment process
# This avoids local Docker build issues with Windows and WSL

echo "Submitting build to Google Cloud Build..."
gcloud builds submit --config=gcloud-build.yaml .

echo "Deployment completed via Google Cloud Build."