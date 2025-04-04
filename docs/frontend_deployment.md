# Frontend Deployment to Google Cloud Run

This guide explains how to deploy the Know Defeat trading dashboard to Google Cloud Run, connecting it to your Cloud SQL database.

## Prerequisites

1. Google Cloud SDK installed and configured
2. Docker installed (for local testing)
3. Node.js and npm for local development
4. Active Google Cloud project with Cloud SQL setup

## Deployment Steps

### 1. Prepare for Deployment

Make sure your database is properly set up in Cloud SQL:

```bash
# Verify database exists
gcloud sql databases list --instance=trading-db
```

### 2. Local Testing

Before deploying to the cloud, test your app locally with:

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

### 3. Deploy to Cloud Run

Use the included deployment script to deploy your app to Cloud Run:

```bash
# Make the script executable if needed
chmod +x deploy-to-cloud-run.sh

# Run the deployment script
./deploy-to-cloud-run.sh
```

The script performs the following steps:
1. Checks Google Cloud authentication
2. Sets the current project
3. Builds the Docker image using Cloud Build
4. Deploys the image to Cloud Run
5. Connects the app to your Cloud SQL instance
6. Displays the public URL when complete

### 4. Modifying Environment Variables

If you need to update environment variables after deployment:

```bash
gcloud run services update know-defeat-frontend \
  --set-env-vars="KEY=VALUE,ANOTHER_KEY=ANOTHER_VALUE"
```

### 5. Setting Up Cloud SQL Proxy for Local Development

For local development with the Cloud SQL database:

```bash
# Run the setup script
./scripts/setup_cloud_sql_proxy.sh
```

This creates a secure tunnel to your Cloud SQL instance on port 5432.

### 6. Common Troubleshooting

- **Connection Error**: Verify your database credentials and ensure the Cloud SQL instance is running
- **Database Access**: Make sure your Cloud Run service has the proper IAM permissions
- **Container Crashes**: Check logs with `gcloud run services logs read know-defeat-frontend`

## Security Considerations

- Never hard-code database passwords in your code
- Use Cloud Secret Manager for sensitive credentials
- Set up proper IAM permissions for your Cloud Run service

## Monitoring

Monitor your deployed application using Cloud Monitoring:

```bash
# Open Cloud Console monitoring for your service
gcloud run services describe know-defeat-frontend --format='value(status.url)'
```

Visit Google Cloud Console to view logs, metrics, and set up alerts.