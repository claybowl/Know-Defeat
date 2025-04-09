// Simple Express server for testing Cloud Run
const express = require('express');
const app = express();
const port = process.env.PORT || 8080;

// Basic endpoints
app.get('/', (req, res) => {
  res.send('Hello from Know-Defeat test server!');
});

app.get('/health', (req, res) => {
  res.send('OK');
});

app.get('/healthcheck', (req, res) => {
  res.send('OK');
});

// Start the server
app.listen(port, '0.0.0.0', () => {
  console.log(`Test server listening on port ${port}`);
});