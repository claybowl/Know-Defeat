#!/bin/bash
# This script builds and deploys the UI directly without using Cloud Build

# Install dependencies with legacy-peer-deps flag
npm install --legacy-peer-deps

# Build the app with legacy-peer-deps
NODE_ENV=production NODE_OPTIONS=--legacy-peer-deps npm run build

# Build the Docker image
docker build -t know-defeat-ui .

# Tag it for Google Container Registry
docker tag know-defeat-ui gcr.io/know-defeat-trading-js/know-defeat-ui:latest

# Push to Google Container Registry
docker push gcr.io/know-defeat-trading-js/know-defeat-ui:latest

# Deploy to Cloud Run
gcloud run deploy know-defeat-ui \
  --image gcr.io/know-defeat-trading-js/know-defeat-ui:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port=80 \
  --set-env-vars VITE_API_URL=https://know-defeat-api-fqag4rwuia-uc.a.run.app/api