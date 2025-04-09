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

# Note: We won't attempt to build the Remix app during container build
# since it requires additional dev dependencies
# The app will run in API-only mode if no build exists

# Set environment variables
ENV PORT=8080
ENV NODE_ENV=production
ENV USE_MOCK_DATA=true

# Verify build directory contents and create a debug file
RUN ls -la /app/build > /app/build-contents.txt || echo "Build directory not found!"
RUN ls -la /app/public > /app/public-contents.txt || echo "Public directory not found!"
RUN find /app -name "*.js" | grep -i remix > /app/remix-files.txt || echo "No Remix files found!"

# Expose port
EXPOSE 8080

# Create startup script with better error reporting
RUN echo '#!/bin/sh\necho "Starting Know-Defeat server..."\nnode server.js\n' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run using the enhanced server (not simple.js anymore)
CMD ["node", "server.js"]