#!/bin/bash
# A very simplified approach that builds locally, packs the build into a simple nginx container, and deploys

echo "Installing dependencies with legacy-peer-deps..."
npm install --legacy-peer-deps

echo "Building React app with legacy-peer-deps..."
NODE_ENV=production NODE_OPTIONS=--legacy-peer-deps npm run build

echo "Creating temporary Dockerfile for serving the built app..."
cat > Dockerfile.temp << 'EOF'
FROM nginx:alpine
COPY dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

echo "Building Docker image..."
docker build -t know-defeat-ui -f Dockerfile.temp .

echo "Tagging for GCR..."
docker tag know-defeat-ui gcr.io/know-defeat-trading-js/know-defeat-ui:latest

echo "Pushing to GCR..."
docker push gcr.io/know-defeat-trading-js/know-defeat-ui:latest

echo "Deploying to Cloud Run..."
gcloud run deploy know-defeat-ui \
  --image gcr.io/know-defeat-trading-js/know-defeat-ui:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port=80

echo "Cleaning up temporary files..."
rm Dockerfile.temp