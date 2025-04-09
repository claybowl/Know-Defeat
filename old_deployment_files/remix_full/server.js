// Combined Express server with both API endpoints and Remix integration
const express = require('express');
const { createRequestHandler } = require('@remix-run/express');
const db = require('./db/database');
const app = express();
const port = process.env.PORT || 8080;

// Set BASE_URL environment variable for server-side API calls
process.env.BASE_URL = process.env.BASE_URL || `http://localhost:${port}`;

// Middleware to parse JSON
app.use(express.json());

// Environment info for diagnostic purposes
console.log('Environment:');
console.log(`  NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`  USE_MOCK_DATA: ${process.env.USE_MOCK_DATA}`);
console.log(`  DB_HOST: ${process.env.DB_HOST || 'not set'}`);
console.log(`  CLOUD_SQL_CONNECTION_NAME: ${process.env.CLOUD_SQL_CONNECTION_NAME || 'not set'}`);
console.log(`  BASE_URL: ${process.env.BASE_URL}`);

// Health check endpoints
app.get('/health', (req, res) => {
  res.send('OK');
});

app.get('/healthcheck', (req, res) => {
  res.send('OK');
});

// API Endpoints
app.get('/api/bots', async (req, res) => {
  try {
    const bots = await db.getBots();
    res.json(bots);
  } catch (error) {
    console.error('Error fetching bots:', error);
    res.status(500).json({ error: 'Failed to fetch bots' });
  }
});

app.get('/api/bots/:id', async (req, res) => {
  try {
    const botId = parseInt(req.params.id);
    const bot = await db.getBotById(botId);
    
    if (!bot) {
      return res.status(404).json({ error: 'Bot not found' });
    }
    
    res.json(bot);
  } catch (error) {
    console.error(`Error fetching bot ${req.params.id}:`, error);
    res.status(500).json({ error: 'Failed to fetch bot details' });
  }
});

app.get('/api/trades', async (req, res) => {
  try {
    const limit = req.query.limit ? parseInt(req.query.limit) : 100;
    const trades = await db.getTrades(limit);
    res.json(trades);
  } catch (error) {
    console.error('Error fetching trades:', error);
    res.status(500).json({ error: 'Failed to fetch trades' });
  }
});

app.get('/api/trades/open', async (req, res) => {
  try {
    const openTrades = await db.getOpenTrades();
    res.json(openTrades);
  } catch (error) {
    console.error('Error fetching open trades:', error);
    res.status(500).json({ error: 'Failed to fetch open trades' });
  }
});

app.get('/api/metrics', async (req, res) => {
  try {
    const metrics = await db.getBotMetrics();
    res.json(metrics);
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).json({ error: 'Failed to fetch metrics' });
  }
});

app.get('/api/dashboard', async (req, res) => {
  try {
    // Fetch data in parallel
    const [bots, openTrades, metrics] = await Promise.all([
      db.getBots(),
      db.getOpenTrades(),
      db.getBotMetrics()
    ]);
    
    // Calculate summary stats
    const totalBots = bots.length;
    const activeBots = bots.filter(bot => bot.is_active).length;
    const totalOpenTrades = openTrades.length;
    
    // Calculate total P&L and average win rate
    const totalPnl = metrics.reduce((sum, bot) => sum + parseFloat(bot.total_pnl || 0), 0);
    const botsWithTrades = metrics.filter(bot => bot.total_trades > 0);
    const avgWinRate = botsWithTrades.length > 0 
      ? botsWithTrades.reduce((sum, bot) => sum + parseFloat(bot.win_rate || 0), 0) / botsWithTrades.length
      : 0;
    
    // Get top performing bots
    const topBots = metrics
      .filter(bot => bot.total_trades > 0)
      .sort((a, b) => parseFloat(b.rank_score || 0) - parseFloat(a.rank_score || 0))
      .slice(0, 5);
    
    // Get recent trades (first 10 from the trades endpoint)
    const recentTrades = await db.getTrades(10);
    
    // Compile dashboard data
    const dashboardData = {
      summary: {
        totalBots,
        activeBots,
        totalOpenTrades,
        totalPnl,
        avgWinRate,
      },
      topBots,
      recentTrades,
      openTrades: openTrades.slice(0, 10), // First 10 open trades
    };
    
    res.json(dashboardData);
  } catch (error) {
    console.error('Error generating dashboard data:', error);
    res.status(500).json({ error: 'Failed to generate dashboard data' });
  }
});

// Serve static files from public directory
app.use(express.static('public'));

// Set up Remix handler for all other routes
try {
  const fs = require('fs');
  const path = require('path');
  
  // Enhanced diagnostics for build files
  console.log("Current working directory:", process.cwd());
  
  const buildPath = path.resolve('./build');
  console.log("Checking for build directory at:", buildPath);
  
  if (!fs.existsSync(buildPath)) {
    console.warn("❌ Build directory not found at:", buildPath);
    throw new Error("Build directory not found");
  }
  
  console.log("✅ Build directory exists. Contents:");
  fs.readdirSync(buildPath).forEach(file => {
    console.log(`  - ${file}`);
  });
  
  const indexPath = path.join(buildPath, 'index.js');
  console.log("Checking for index.js at:", indexPath);
  
  if (!fs.existsSync(indexPath)) {
    console.warn("❌ index.js not found at:", indexPath);
    throw new Error("index.js not found");
  }
  
  console.log("✅ index.js exists. Size:", fs.statSync(indexPath).size, "bytes");
  
  // Check if we can load the build module
  console.log("Attempting to load build module...");
  try {
    // Try with direct path first
    const build = require("./build");
    console.log("✅ Build module loaded successfully:", Object.keys(build));
  } catch (buildError) {
    console.error("Failed to load build with ./build:", buildError.message);
    
    try {
      // Try with absolute path
      const absPath = path.resolve('./build');
      console.log("Trying absolute path import:", absPath);
      const build = require(absPath);
      console.log("✅ Build module loaded successfully with absolute path:", Object.keys(build));
    } catch (absError) {
      console.error("Failed to load build with absolute path:", absError.message);
      throw new Error("Could not load build module");
    }
  }
  
  // Get the correct build module reference
  let buildModule;
  try {
    buildModule = require("./build");
  } catch (err) {
    try {
      buildModule = require(path.resolve('./build'));
    } catch (absErr) {
      console.error("Could not load build module by any means");
      throw new Error("Build module not available");
    }
  }
  
  app.all(
    "*",
    createRequestHandler({
      build: buildModule,
      mode: process.env.NODE_ENV,
    })
  );
  console.log("✅ Remix request handler set up successfully");
} catch (error) {
  console.warn("Unable to set up Remix request handler:", error.message);
  console.warn("Falling back to simple API server");
  
  // Fallback route for non-API requests when Remix build isn't available
  app.get('*', (req, res) => {
    if (req.url.startsWith('/api/')) {
      res.status(404).json({ error: 'API endpoint not found' });
    } else {
      res.send(`
        <!DOCTYPE html>
        <html>
          <head>
            <title>Know-Defeat Trading System</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
              }
              h1 { color: #2c5282; }
              .card {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              }
              .links {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
              }
              .links a {
                background: #4299e1;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;
              }
            </style>
          </head>
          <body>
            <h1>Know-Defeat Trading System</h1>
            
            <div class="card">
              <h2>System Status</h2>
              <p>The API server is <strong>operational</strong>.</p>
              <p>Remix UI is <strong>not yet available</strong>. Try running <code>npm run build</code> first.</p>
              <p><em>Last updated: ${new Date().toLocaleString()}</em></p>
            </div>
            
            <div class="links">
              <a href="/api/bots">View Bots</a>
              <a href="/api/trades">View Trades</a>
              <a href="/api/metrics">View Metrics</a>
              <a href="/api/dashboard">Dashboard Data</a>
            </div>
          </body>
        </html>
      `);
    }
  });
}

// Explicitly handle startup errors
console.log(`Attempting to start server on port ${port}...`);

const server = app.listen(port, '0.0.0.0', () => {
  console.log(`✅ Know-Defeat enhanced server running on port ${port}`);
  console.log(`Server is listening at http://0.0.0.0:${port}`);
});

// Handle server errors
server.on('error', (error) => {
  console.error('SERVER ERROR:', error);
  
  if (error.code === 'EADDRINUSE') {
    console.error(`Port ${port} is already in use. Choose another port.`);
  }
  
  // Exit with error code for visibility
  process.exit(1);
});

// Log when process is about to exit
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
  });
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('UNCAUGHT EXCEPTION:', error);
  process.exit(1);
});