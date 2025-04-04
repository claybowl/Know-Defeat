// Simple Express server for hosting the Remix app in production
const express = require('express');
const { createRequestHandler } = require('@remix-run/express');
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;

// Serve static files
app.use(express.static('public'));
app.use(express.static('build/client'));

// Remix request handler
app.all(
  '*',
  createRequestHandler({
    build: path.join(process.cwd(), 'build'),
  })
);

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});