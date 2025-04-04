#!/bin/bash
# Deploy the Know Defeat Frontend to Google Cloud Run

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-frontend"
DB_USER="postgres"
DB_NAME="tick_data"
CLOUD_SQL_INSTANCE="trading-db"
CLOUD_SQL_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

# Ensure Google Cloud SDK is authenticated
echo "Checking Google Cloud authentication..."
gcloud auth list

# Set project ID
echo "Setting project to: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# Ask for database password
echo -n "Enter your database password (input will be hidden): "
read -s DB_PASSWORD
echo ""  # Add newline after password input

# Check if DB_PASSWORD is empty
if [ -z "$DB_PASSWORD" ]; then
  echo "Error: Database password cannot be empty"
  exit 1
fi

# Build Docker image
echo "Building Docker image..."
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Check if a secret for DB password exists, if not create it
if ! gcloud secrets describe db-password &>/dev/null; then
  echo "Creating database password secret..."
  echo -n "$DB_PASSWORD" | gcloud secrets create db-password --data-file=-
else
  echo "Updating database password secret..."
  echo -n "$DB_PASSWORD" | gcloud secrets versions add db-password --data-file=-
fi

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --add-cloudsql-instances ${CLOUD_SQL_CONNECTION_NAME} \
  --set-env-vars "DB_USER=${DB_USER},DB_NAME=${DB_NAME},CLOUD_SQL_CONNECTION_NAME=${CLOUD_SQL_CONNECTION_NAME},DB_HOST=/cloudsql,USE_MOCK_DATA=false" \
  --update-secrets=DB_PASSWORD=db-password:latest

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Deployment complete!"
echo "Your application is available at: ${SERVICE_URL}"