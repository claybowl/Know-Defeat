#!/bin/bash
# Deploy the UI to Google Cloud Run

# Build the Docker image
docker build -t know-defeat-ui .

# Tag it for Google Artifact Registry
docker tag know-defeat-ui gcr.io/know-defeat-trading-js/know-defeat-ui:latest

# Push to Google Container Registry
docker push gcr.io/know-defeat-trading-js/know-defeat-ui:latest

# Deploy to Cloud Run
gcloud run deploy know-defeat-ui \
  --image gcr.io/know-defeat-trading-js/know-defeat-ui:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port=80