// Enhanced server with API endpoints but keeping the core structure
const http = require('http');
const url = require('url');
const db = require('./db/database');

const port = process.env.PORT || 8080;

// Environment info for diagnostic purposes
console.log('Environment:');
console.log(`  NODE_ENV: ${process.env.NODE_ENV}`);
console.log(`  USE_MOCK_DATA: ${process.env.USE_MOCK_DATA}`);
console.log(`  PORT: ${port}`);

// Helper to send JSON response
function sendJSON(res, data, status = 200) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(data));
}

// Helper to send HTML response
function sendHTML(res, html, status = 200) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'text/html');
  res.end(html);
}

// API route handlers
const apiHandlers = {
  // Get all bots
  '/api/bots': async (req, res) => {
    try {
      const bots = await db.getBots();
      sendJSON(res, bots);
    } catch (error) {
      console.error('Error in /api/bots:', error);
      sendJSON(res, { error: 'Failed to fetch bots' }, 500);
    }
  },
  
  // Get bot by ID (extract ID from path like /api/bots/1)
  '/api/bots/': async (req, res, pathParts) => {
    try {
      const botId = parseInt(pathParts[3]);
      if (isNaN(botId)) {
        sendJSON(res, { error: 'Invalid bot ID' }, 400);
        return;
      }
      
      const bot = await db.getBotById(botId);
      if (!bot) {
        sendJSON(res, { error: 'Bot not found' }, 404);
        return;
      }
      
      sendJSON(res, bot);
    } catch (error) {
      console.error(`Error in /api/bots/:id:`, error);
      sendJSON(res, { error: 'Failed to fetch bot details' }, 500);
    }
  },
  
  // Get all trades
  '/api/trades': async (req, res, pathParts, query) => {
    try {
      const limit = query.limit ? parseInt(query.limit) : 100;
      const trades = await db.getTrades(limit);
      sendJSON(res, trades);
    } catch (error) {
      console.error('Error in /api/trades:', error);
      sendJSON(res, { error: 'Failed to fetch trades' }, 500);
    }
  },
  
  // Get open trades
  '/api/trades/open': async (req, res) => {
    try {
      const openTrades = await db.getOpenTrades();
      sendJSON(res, openTrades);
    } catch (error) {
      console.error('Error in /api/trades/open:', error);
      sendJSON(res, { error: 'Failed to fetch open trades' }, 500);
    }
  },
  
  // Get bot metrics
  '/api/metrics': async (req, res) => {
    try {
      const metrics = await db.getBotMetrics();
      sendJSON(res, metrics);
    } catch (error) {
      console.error('Error in /api/metrics:', error);
      sendJSON(res, { error: 'Failed to fetch metrics' }, 500);
    }
  },
  
  // Get dashboard data
  '/api/dashboard': async (req, res) => {
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
      
      // Get recent trades
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
      
      sendJSON(res, dashboardData);
    } catch (error) {
      console.error('Error in /api/dashboard:', error);
      sendJSON(res, { error: 'Failed to generate dashboard data' }, 500);
    }
  }
};

// Create a basic HTTP server
const server = http.createServer(async (req, res) => {
  try {
    console.log(`Received request for ${req.url}`);
    
    // Parse the URL and query parameters
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;
    const pathParts = pathname.split('/');
    const query = parsedUrl.query;
    
    // Handle health check endpoints
    if (pathname === '/health' || pathname === '/healthcheck') {
      res.statusCode = 200;
      res.setHeader('Content-Type', 'text/plain');
      res.end('OK');
      return;
    }
    
    // Handle API routes
    if (pathname.startsWith('/api/')) {
      // Check for exact route matches
      if (apiHandlers[pathname]) {
        await apiHandlers[pathname](req, res, pathParts, query);
        return;
      }
      
      // Check for parameterized routes (e.g., /api/bots/1)
      if (pathParts.length >= 4 && pathname.startsWith('/api/bots/')) {
        await apiHandlers['/api/bots/'](req, res, pathParts, query);
        return;
      }
      
      // Handle unknown API routes
      sendJSON(res, { error: 'API endpoint not found' }, 404);
      return;
    }
    
    // Handle the main HTML page
    if (pathname === '/') {
      // HTML template with improved styling
      sendHTML(res, `
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
              <p>The Know-Defeat trading system API is <strong>operational</strong>.</p>
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
                    card.innerHTML = \`
                      <div class="stat-value">\${stat.value}</div>
                      <div class="stat-label">\${stat.label}</div>
                    \`;
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
      return;
    }
    
    // Handle 404 for unknown routes
    res.statusCode = 404;
    res.setHeader('Content-Type', 'text/html');
    res.end(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>404 - Not Found</title>
          <meta charset="utf-8">
        </head>
        <body>
          <h1>404 - Not Found</h1>
          <p>The resource you are looking for does not exist.</p>
          <p><a href="/">Go back to home page</a></p>
        </body>
      </html>
    `);
  } catch (error) {
    console.error('Unhandled request error:', error);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'text/plain');
    res.end('Internal Server Error');
  }
});

// Start the server and log the process
console.log(`Starting server on port ${port}...`);

server.listen(port, '0.0.0.0', () => {
  console.log(`✅ Know-Defeat enhanced server running at http://0.0.0.0:${port}/`);
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