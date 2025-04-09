#!/bin/bash
# Script to deploy just the UI component using pre-built container

# Configuration
PROJECT_ID="know-defeat-trading-js"
REGION="us-central1"
UI_SERVICE="know-defeat-ui"
API_URL="https://know-defeat-api-fqag4rwuia-uc.a.run.app"

# Ensure we're using the right project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

echo "Building and deploying a simplified UI container..."

# Navigate to the UI directory
cd ../ui

# Create a simple Dockerfile for deployment only
cat > Dockerfile.simple << 'EOF'
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN mkdir -p /usr/share/nginx/html
COPY index.html /usr/share/nginx/html/
ENV VITE_API_URL="https://know-defeat-api-fqag4rwuia-uc.a.run.app/api"
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

# Create a simple index.html that redirects to API
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Know Defeat Trading System</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      background-color: #f5f5f5;
      color: #333;
    }
    .container {
      text-align: center;
      padding: 2rem;
      background-color: white;
      border-radius: 10px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      max-width: 800px;
    }
    h1 {
      color: #1976d2;
    }
    .links {
      margin-top: 2rem;
    }
    .api-link {
      display: inline-block;
      padding: 0.75rem 1.5rem;
      margin: 0.5rem;
      background-color: #1976d2;
      color: white;
      text-decoration: none;
      border-radius: 4px;
      font-weight: bold;
      transition: background-color 0.3s;
    }
    .api-link:hover {
      background-color: #1565c0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Know Defeat Trading System</h1>
    <p>Welcome to the Know Defeat Trading System. The full UI is still in development, but you can access the API directly.</p>
    
    <div class="links">
      <a href="https://know-defeat-api-fqag4rwuia-uc.a.run.app/api/dashboard" class="api-link">Access Dashboard API</a>
      <a href="https://know-defeat-api-fqag4rwuia-uc.a.run.app/api/bots" class="api-link">View Bots API</a>
      <a href="https://know-defeat-api-fqag4rwuia-uc.a.run.app/api/trades" class="api-link">View Trades API</a>
    </div>
    
    <p style="margin-top: 2rem; color: #666;">
      The data is presented in JSON format. You can use browser extensions like JSON Viewer to better visualize the data.
    </p>
  </div>
</body>
</html>
EOF

# Build and deploy a simple container
echo "Building and pushing UI container..."
# First create a temporary directory for the build
mkdir -p temp_build
cp index.html temp_build/
cp nginx.conf temp_build/
cp Dockerfile.simple temp_build/Dockerfile
cd temp_build

# Submit build from the temp directory
echo "Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/know-defeat-ui

# Return to ui directory
cd ..

echo "Deploying UI to Cloud Run..."
gcloud run deploy $UI_SERVICE \
  --image gcr.io/$PROJECT_ID/know-defeat-ui \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 80

# Get the UI service URL
echo "Getting UI service URL..."
UI_URL=$(gcloud run services describe $UI_SERVICE --platform managed --region $REGION --format 'value(status.url)')

echo "==============================================="
echo "UI Deployment Complete!"
echo "UI URL: $UI_URL"
echo "==============================================="