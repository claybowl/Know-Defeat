// Local development server with hot reloading for API
const express = require('express');
const db = require('./db/database');
const app = express();
const port = process.env.PORT || 3000;

// Set BASE_URL for server-side API calls
process.env.BASE_URL = process.env.BASE_URL || `http://localhost:${port}`;

// Middleware to parse JSON
app.use(express.json());

// Add CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

// Log all requests for debugging
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

// Environment info for diagnostic purposes
console.log('Environment:');
console.log(`  NODE_ENV: ${process.env.NODE_ENV || 'development'}`);
console.log(`  USE_MOCK_DATA: ${process.env.USE_MOCK_DATA || 'true'}`);
console.log(`  PORT: ${port}`);

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

// Basic homepage for development
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
      <head>
        <title>Know-Defeat API Development Server</title>
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
          code {
            background: #edf2f7;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: monospace;
          }
        </style>
      </head>
      <body>
        <h1>Know-Defeat API Development Server</h1>
        
        <div class="card">
          <h2>Development Mode</h2>
          <p>The API server is running in <strong>development mode</strong>.</p>
          <p><em>Last updated: ${new Date().toLocaleString()}</em></p>
        </div>
        
        <div class="card">
          <h2>Available API Endpoints</h2>
          <ul>
            <li><code>GET /api/bots</code> - List all trading bots</li>
            <li><code>GET /api/bots/:id</code> - Get details for a specific bot</li>
            <li><code>GET /api/trades</code> - List recent trades</li>
            <li><code>GET /api/trades/open</code> - List open trades</li>
            <li><code>GET /api/metrics</code> - Get performance metrics</li>
            <li><code>GET /api/dashboard</code> - Get dashboard summary data</li>
          </ul>
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
});

// Start the server and log the process
console.log(`Starting development server on port ${port}...`);

const server = app.listen(port, '0.0.0.0', () => {
  console.log(`✅ Know-Defeat development server running at http://0.0.0.0:${port}/`);
  console.log('Press Ctrl+C to stop');
});

// Handle server errors
server.on('error', (error) => {
  console.error('SERVER ERROR:', error);
  
  if (error.code === 'EADDRINUSE') {
    console.error(`Port ${port} is already in use. Choose another port.`);
  }
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down development server...');
  server.close(() => {
    console.log('Server stopped');
    process.exit(0);
  });
});