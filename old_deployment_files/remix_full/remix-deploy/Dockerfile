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
RUN echo '#!/bin/sh' > /app/debug.sh && \
    echo 'echo "=== Build Directory Contents ===" && ls -la /app/build' >> /app/debug.sh && \
    echo 'echo "=== Public Directory Contents ===" && ls -la /app/public' >> /app/debug.sh && \
    echo 'echo "=== Server Module Details ===" && node -e "console.log(require.resolve(\"./server.js\"))"' >> /app/debug.sh && \
    echo 'echo "=== Attempting to load build ===" && node -e "try { const build = require(\"./build\"); console.log(Object.keys(build)); } catch (e) { console.error(e); }"' >> /app/debug.sh && \
    chmod +x /app/debug.sh

# Set environment variables
ENV PORT=8080
ENV NODE_ENV=production
ENV USE_MOCK_DATA=true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run debug script and then server
CMD ["/bin/sh", "-c", "/app/debug.sh && node server.js"]
