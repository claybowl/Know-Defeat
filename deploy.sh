#!/bin/bash
# Master deployment script for Know-Defeat

# Which component to deploy?
# Options: remix-minimal, test-app, full
COMPONENT=${1:-"remix-minimal"}

echo "Deploying component: $COMPONENT"

case $COMPONENT in
  "remix-minimal")
    echo "Deploying minimal Remix app..."
    cd remix-minimal
    ./deploy.sh
    cd ..
    ;;
  "test-app")
    echo "Deploying test Express app..."
    cd test-app
    ./deploy.sh
    cd ..
    ;;
  "full")
    echo "This option is for deploying the full app once it's ready"
    echo "Currently not implemented - please use remix-minimal first"
    ;;
  *)
    echo "Unknown component: $COMPONENT"
    echo "Valid options: remix-minimal, test-app, full"
    exit 1
    ;;
esac

echo "Deployment complete!"
echo "See strategy document for next steps: cloud-deploy-strategy.md"