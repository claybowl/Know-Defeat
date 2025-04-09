// Enhanced Express server with real data API endpoints
const express = require('express');
const db = require('./db/database');
const app = express();
const port = process.env.PORT || 8080;

// Middleware to parse JSON
app.use(express.json());

// Environment info for diagnostic purposes
console.log('Environment:');
console.log(`  NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`  USE_MOCK_DATA: ${process.env.USE_MOCK_DATA}`);
console.log(`  DB_HOST: ${process.env.DB_HOST || 'not set'}`);
console.log(`  CLOUD_SQL_CONNECTION_NAME: ${process.env.CLOUD_SQL_CONNECTION_NAME || 'not set'}`);

// Basic HTML homepage
app.get('/', (req, res) => {
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
          h1 {
            color: #2c5282;
          }
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
          .links a:hover {
            background: #3182ce;
          }
          code {
            background: #edf2f7;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: monospace;
          }
          .stats {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 20px;
          }
          .stat-card {
            background: white;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
          }
          .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2b6cb0;
            margin: 8px 0;
          }
          .stat-label {
            font-size: 14px;
            color: #718096;
          }
        </style>
      </head>
      <body>
        <h1>Know-Defeat Trading System</h1>
        
        <div class="card">
          <h2>System Status</h2>
          <p>The Know-Defeat trading system is <strong>operational</strong>.</p>
          <p><em>Last updated: ${new Date().toLocaleString()}</em></p>
        </div>
        
        <div class="links">
          <a href="/api/bots">View Bots</a>
          <a href="/api/trades">View Trades</a>
          <a href="/api/metrics">View Metrics</a>
          <a href="/api/dashboard">Dashboard Data</a>
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
        
        <div id="stats-container">
          <h2>Loading Stats...</h2>
        </div>
        
        <script>
          // Fetch dashboard data
          fetch('/api/dashboard')
            .then(response => response.json())
            .then(data => {
              const statsContainer = document.getElementById('stats-container');
              statsContainer.innerHTML = '<h2>System Overview</h2><div class="stats"></div>';
              const statsGrid = statsContainer.querySelector('.stats');
              
              // Create stats cards
              const stats = [
                { label: 'Total Bots', value: data.summary.totalBots },
                { label: 'Active Bots', value: data.summary.activeBots },
                { label: 'Open Trades', value: data.summary.totalOpenTrades },
                { label: 'Total P&L', value: '$' + data.summary.totalPnl.toLocaleString(undefined, {maximumFractionDigits: 2}) },
                { label: 'Avg Win Rate', value: (data.summary.avgWinRate * 100).toFixed(1) + '%' }
              ];
              
              stats.forEach(stat => {
                const card = document.createElement('div');
                card.className = 'stat-card';
                card.innerHTML = `
                  <div class="stat-value">${stat.value}</div>
                  <div class="stat-label">${stat.label}</div>
                `;
                statsGrid.appendChild(card);
              });
            })
            .catch(error => {
              console.error('Error fetching data:', error);
              document.getElementById('stats-container').innerHTML = '<h2>Stats Unavailable</h2><p>Could not load system statistics.</p>';
            });
        </script>
      </body>
    </html>
  `);
});

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