// Express server with Remix handler for Cloud Run
const express = require('express');
const { createRequestHandler } = require('@remix-run/express');
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;

// Server options for Remix handler
const BUILD_DIR = path.join(process.cwd(), "build");

// Log startup info
console.log('Starting server...');
console.log(`  Build directory: ${BUILD_DIR}`);
console.log(`  Port: ${port}`);

try {
  // Serve static files
  app.use(express.static('public'));
  app.use(express.static('build/client'));

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.status(200).send('OK');
  });

  // Log environment variables (without sensitive info)
  console.log('Environment:');
  console.log(`  NODE_ENV: ${process.env.NODE_ENV}`);
  console.log(`  PORT: ${process.env.PORT}`);
  console.log(`  USE_MOCK_DATA: ${process.env.USE_MOCK_DATA}`);

  // This handles all routes
  app.all(
    '*',
    createRequestHandler({
      build: BUILD_DIR,
      mode: process.env.NODE_ENV,
    })
  );

  // Start server
  app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${port}/`);
  });
} catch (error) {
  console.error('Error starting server:', error);
}