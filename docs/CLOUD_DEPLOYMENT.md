# Know-Defeat Cloud Deployment Guide

This document outlines how to deploy the Know-Defeat application to Google Cloud Run.

## Available Deployment Options

We have created a staged deployment approach to help you get your application running in the cloud:

1. **Test App**: A simple Express server to validate your GCP setup
2. **Remix Minimal**: A simplified version of your application with mock data
3. **Full App**: The complete Remix application (future state)

## How to Deploy

Use the master deployment script to deploy any component:

```bash
# Deploy the minimal Remix app (default)
./deploy.sh

# Deploy the test app
./deploy.sh test-app

# Deploy the full app (when ready)
./deploy.sh full
```

## Deployment Strategy

For details on the deployment strategy, please see [cloud-deploy-strategy.md](cloud-deploy-strategy.md).

## File Structure

```
/Know-Defeat/
  ├── deploy.sh                 # Master deployment script
  ├── cloud-deploy-strategy.md  # Detailed strategy document
  ├── CLOUD_DEPLOYMENT.md       # This file
  ├── remix-minimal/            # Simplified Remix app for deployment
  │   ├── deploy.sh             # Deploy script for minimal app
  │   ├── Dockerfile            # Docker configuration
  │   ├── package.json          # Minimal dependencies
  │   └── server.js             # Simple Express server
  ├── test-app/                 # Test Express app
  │   ├── deploy.sh             # Deploy script for test app
  │   ├── Dockerfile            # Docker configuration
  │   ├── package.json          # Express dependencies
  │   └── server.js             # Simple test server
  └── old_deployment_files/     # Archive of old deployment files
```

## Troubleshooting

If you encounter deployment issues:

1. Start with the test app to verify your GCP setup is correct
2. Try the minimal Remix app to validate the basics of your application
3. Review the deployment logs by running:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=know-defeat-remix-minimal" --limit 50
   ```

## Next Steps

After successfully deploying the minimal app, follow the migration plan in the strategy document to gradually add functionality.