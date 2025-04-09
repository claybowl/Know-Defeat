// Express server with Remix handler for Cloud Run
const express = require('express');
const { createRequestHandler } = require('@remix-run/express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 8080;

// Server options for Remix handler
const BUILD_DIR = path.join(process.cwd(), "build");

// Log startup info
console.log('Starting server...');
console.log(`  Build directory: ${BUILD_DIR}`);
console.log(`  Port: ${port}`);

// Verify build directory and contents
try {
  const buildExists = fs.existsSync(BUILD_DIR);
  console.log(`  Build directory exists: ${buildExists}`);
  
  if (buildExists) {
    const buildFiles = fs.readdirSync(BUILD_DIR);
    console.log(`  Build directory contents: ${buildFiles.join(', ')}`);
    
    // Check for index.js specifically
    const indexPath = path.join(BUILD_DIR, 'index.js');
    console.log(`  index.js exists: ${fs.existsSync(indexPath)}`);
  }
} catch (error) {
  console.error('Error checking build directory:', error);
}

try {
  // Serve static files
  app.use(express.static('public'));
  app.use(express.static('build/client'));

  // Add explicit error handler
  app.use((err, req, res, next) => {
    console.error('Express error:', err);
    res.status(500).send(`Internal Server Error: ${err.message}`);
  });

  // Health check endpoint
  app.get('/health', (req, res) => {
    res.status(200).send('OK');
  });

  app.get('/healthcheck', (req, res) => {
    res.status(200).send('OK');
  });

  // Log environment variables (without sensitive info)
  console.log('Environment:');
  console.log(`  NODE_ENV: ${process.env.NODE_ENV}`);
  console.log(`  PORT: ${process.env.PORT}`);
  console.log(`  USE_MOCK_DATA: ${process.env.USE_MOCK_DATA}`);

  // Handle Remix routes with better error handling
  try {
    // This handles all routes
    app.all(
      '*',
      (req, res, next) => {
        // Simple middleware to catch and log any errors
        try {
          return createRequestHandler({
            build: BUILD_DIR,
            mode: process.env.NODE_ENV,
          })(req, res, next);
        } catch (error) {
          console.error('Remix handler error:', error);
          next(error);
        }
      }
    );
  } catch (error) {
    console.error('Error creating request handler:', error);
  }

  // Start server
  app.listen(port, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${port}/`);
  });
} catch (error) {
  console.error('Error details:', {
    message: error.message,
    stack: error.stack,
    name: error.name
  });
}