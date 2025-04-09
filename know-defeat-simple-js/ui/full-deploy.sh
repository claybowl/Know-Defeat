#!/bin/bash
# A comprehensive script to deploy the full React UI

echo "===== STEP 1: Setting up environment ====="
# Make sure we're in the right directory
cd "$(dirname "$0")"

echo "===== STEP 2: Cleaning up and installing dependencies ====="
# Remove old build files
rm -rf dist
rm -f Dockerfile.temp

# Install dependencies
npm install --legacy-peer-deps

echo "===== STEP 3: Building the React application ====="
# Build the React app
npm run build --legacy-peer-deps

echo "===== STEP 4: Checking build output ====="
ls -la dist
ls -la dist/assets

echo "===== STEP 5: Creating Docker image ====="
# Create a temporary Dockerfile for the build
cat > Dockerfile.temp << 'EOF'
FROM nginx:alpine

# Copy built files to the nginx html directory
COPY dist/ /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Set API URL environment variable if needed
ENV VITE_API_URL="https://know-defeat-api-fqag4rwuia-uc.a.run.app/api"

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
EOF

# Build the Docker image
docker build -t know-defeat-ui-full -f Dockerfile.temp .

echo "===== STEP 6: Tagging and pushing to Google Container Registry ====="
# Tag for Google Container Registry
docker tag know-defeat-ui-full gcr.io/know-defeat-trading-js/know-defeat-ui:latest

# Push to Google Container Registry
docker push gcr.io/know-defeat-trading-js/know-defeat-ui:latest

echo "===== STEP 7: Deploying to Cloud Run ====="
# Deploy to Cloud Run
gcloud run deploy know-defeat-ui \
  --image gcr.io/know-defeat-trading-js/know-defeat-ui:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port=80

echo "===== STEP 8: Cleaning up ====="
rm Dockerfile.temp

echo "===== Deployment Complete! ====="
echo "Your full React UI should now be available at the Cloud Run URL above."