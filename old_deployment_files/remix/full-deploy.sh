#!/bin/bash
# Full deployment script that builds the UI and deploys to Cloud Run

# Exit on error
set -e

# Color constants for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_status() {
  echo -e "${2}${1}${NC}"
}

# Print step info
print_step() {
  echo -e "\n${YELLOW}== $1 ==${NC}"
}

print_step "Setting up environment for full UI deployment"

# 1. Set up development environment
print_status "Installing development dependencies..." "${YELLOW}"
npm install -D @remix-run/dev @types/react @types/react-dom typescript

# 2. Build the Remix application
print_step "Building Remix application"
npm run build

# 3. Create a deployment directory
print_step "Creating deployment package"
DEPLOY_DIR="remix-deploy"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# 4. Copy necessary files
print_status "Copying files to deployment directory..." "${GREEN}"
cp -r app $DEPLOY_DIR/
cp -r build $DEPLOY_DIR/
cp -r db $DEPLOY_DIR/
cp -r public $DEPLOY_DIR/
cp package.json $DEPLOY_DIR/
cp server.js $DEPLOY_DIR/
cp Dockerfile $DEPLOY_DIR/

# Force the build directory to be included in the deployment
mkdir -p $DEPLOY_DIR/build
cp -r build/* $DEPLOY_DIR/build/

# 5. Create a minimal package.json without dev dependencies
cat > $DEPLOY_DIR/package.json << EOF
{
  "name": "know-defeat-remix-minimal",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "@remix-run/express": "^2.8.1",
    "@remix-run/node": "^2.8.1",
    "@remix-run/react": "^2.8.1",
    "express": "^4.18.2",
    "pg": "^8.11.3",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
EOF

# 6. Modify Dockerfile to skip build step
cat > $DEPLOY_DIR/Dockerfile << EOF
FROM node:18-alpine

# Install debugging utilities and dependencies for Cloud SQL proxy
RUN apk add --no-cache curl procps tini postgresql-client

WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Add debug script to help troubleshoot
RUN echo '#!/bin/sh' > /app/debug.sh && \\
    echo 'echo "=== Build Directory Contents ===" && ls -la /app/build' >> /app/debug.sh && \\
    echo 'echo "=== Public Directory Contents ===" && ls -la /app/public' >> /app/debug.sh && \\
    echo 'echo "=== Server Module Details ===" && node -e "console.log(require.resolve(\\"./server.js\\"))"' >> /app/debug.sh && \\
    echo 'echo "=== Attempting to load build ===" && node -e "try { const build = require(\\"./build\\"); console.log(Object.keys(build)); } catch (e) { console.error(e); }"' >> /app/debug.sh && \\
    chmod +x /app/debug.sh

# Set environment variables
ENV PORT=8080
ENV NODE_ENV=production
ENV USE_MOCK_DATA=true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8080/health || exit 1

# Run debug script and then server
CMD ["/bin/sh", "-c", "/app/debug.sh && node server.js"]
EOF

# 7. Create simplified deploy script
cat > $DEPLOY_DIR/deploy.sh << EOF
#!/bin/bash

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-ui"

# Set project ID
gcloud config set project \${PROJECT_ID}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy \${SERVICE_NAME} \\
  --source . \\
  --platform managed \\
  --region \${REGION} \\
  --allow-unauthenticated \\
  --set-env-vars "NODE_ENV=production,USE_MOCK_DATA=true" \\
  --max-instances=2 \\
  --memory=512Mi \\
  --timeout=30s
EOF
chmod +x $DEPLOY_DIR/deploy.sh

# 8. Deploy from the deployment directory
print_step "Deploying to Cloud Run"
cd $DEPLOY_DIR
./deploy.sh

# 9. Clean up
print_step "Deployment completed"
cd ..
print_status "You can find the deployment files in the $DEPLOY_DIR directory" "${GREEN}"
print_status "Your UI should now be available at the URL shown above" "${GREEN}"}