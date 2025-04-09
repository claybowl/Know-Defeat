FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files first for better layer caching
COPY package.json package-lock.json* ./
RUN npm ci

# Copy the rest of the application
COPY . .

# Build the app with CJS format
ENV NODE_ENV=production
RUN npm run build

# Set runtime environment
ENV PORT=8080
EXPOSE 8080

# Use mock data by default (change in deployment)
ENV USE_MOCK_DATA=true
ENV NODE_ENV=production

# Create a simple express server for healthcheck
RUN echo "const express = require('express'); \
  const app = express(); \
  app.get('/health', (req, res) => res.send('OK')); \
  app.get('/healthcheck', (req, res) => res.send('OK')); \
  app.listen(8080, () => console.log('Health server running'));" > health-server.js

# Create a combined run script that ensures healthcheck runs
RUN echo "#!/bin/sh \
  \n# Start health server in background \
  \nnode health-server.js & \
  \n# Start main app \
  \nnode server.js" > start.sh && chmod +x start.sh

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost:8080/healthcheck || exit 1

# Run the app
CMD ["./start.sh"]