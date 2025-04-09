// Simple health check server
const express = require('express');
const app = express();
const port = process.env.HEALTH_PORT || 8081;

app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

app.get('/healthcheck', (req, res) => {
  res.status(200).send('OK');
});

// Start server
app.listen(port, '0.0.0.0', () => {
  console.log(`Health check server running on port ${port}`);
});