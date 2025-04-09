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

# Set environment variables
ENV PORT=8080
ENV NODE_ENV=production
ENV USE_MOCK_DATA=true

# Expose port
EXPOSE 8080

# Create startup script for better error handling
RUN echo '#!/bin/sh\necho "Starting server..."\nnode server.js\n' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run using the enhanced API server
CMD ["node", "simple.js"]