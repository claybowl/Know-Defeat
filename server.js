// Express server for hosting the Remix app in production
const express = require('express');
const { createRequestHandler } = require('@remix-run/express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 8080;

// Log environment
console.log('Environment:');
console.log('  NODE_ENV:', process.env.NODE_ENV);
console.log('  PORT:', port);
console.log('  USE_MOCK_DATA:', process.env.USE_MOCK_DATA);

// Explicit health check endpoints
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

app.get('/healthcheck', (req, res) => {
  res.status(200).send('OK');
});

// Handle static files
console.log('Setting up static files...');
app.use(express.static('public'));
app.use(express.static('build/client'));

// Add error handler middleware
app.use((err, req, res, next) => {
  console.error('Express error:', err);
  res.status(500).send('Internal Server Error');
});

// Check if build directory exists
const buildDir = path.join(process.cwd(), 'build');
console.log(`Checking build directory: ${buildDir}`);
if (fs.existsSync(buildDir)) {
  console.log('Build directory exists');
  const files = fs.readdirSync(buildDir);
  console.log('Files in build directory:', files);
} else {
  console.error('Build directory does not exist!');
}

// Create Remix request handler with error handling
console.log('Setting up Remix handler...');
try {
  app.all(
    '*',
    (req, res, next) => {
      // Special handling for the root to make debugging easier
      if (req.path === '/') {
        console.log('Handling root request');
      }
      
      try {
        return createRequestHandler({
          build: buildDir,
          mode: process.env.NODE_ENV
        })(req, res, next);
      } catch (err) {
        console.error('Remix handler error:', err);
        next(err);
      }
    }
  );
} catch (err) {
  console.error('Failed to create request handler:', err);
}

// Start the server
console.log('Starting server...');
app.listen(port, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${port}`);
}).on('error', (err) => {
  console.error('Server failed to start:', err);
});