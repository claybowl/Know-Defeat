#!/bin/bash
# Script to deploy a simple static HTML page to Cloud Run

# Configuration
PROJECT_ID="know-defeat-trading-js"
REGION="us-central1"
UI_SERVICE="know-defeat-ui"
API_URL="https://know-defeat-api-fqag4rwuia-uc.a.run.app"

# Ensure we're using the right project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Create a directory for our static site
echo "Creating static site..."
mkdir -p static-site
cd static-site

# Create a simple index.html
cat > index.html << EOF
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
      <a href="${API_URL}/api/dashboard" class="api-link">Access Dashboard API</a>
      <a href="${API_URL}/api/bots" class="api-link">View Bots API</a>
      <a href="${API_URL}/api/trades" class="api-link">View Trades API</a>
    </div>
    
    <p style="margin-top: 2rem; color: #666;">
      The data is presented in JSON format. You can use browser extensions like JSON Viewer to better visualize the data.
    </p>
  </div>
</body>
</html>
EOF

# Create a simple Dockerfile
cat > Dockerfile << EOF
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
EOF

# Create nginx config
cat > nginx.conf << EOF
server {
    listen 8080;
    server_name localhost;
    location / {
        root /usr/share/nginx/html;
        index index.html;
    }
}
EOF

# Build and push the Docker image
echo "Building and pushing the Docker image..."
docker build -t gcr.io/$PROJECT_ID/$UI_SERVICE .
docker push gcr.io/$PROJECT_ID/$UI_SERVICE

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $UI_SERVICE \
  --image gcr.io/$PROJECT_ID/$UI_SERVICE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated

# Get the UI service URL
echo "Getting UI service URL..."
UI_URL=$(gcloud run services describe $UI_SERVICE --platform managed --region $REGION --format 'value(status.url)')

echo "==============================================="
echo "UI Deployment Complete!"
echo "UI URL: $UI_URL"
echo "==============================================="

# Clean up
cd ..
rm -rf static-site