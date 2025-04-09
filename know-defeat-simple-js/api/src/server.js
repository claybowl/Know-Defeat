const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const dotenv = require('dotenv');
const { Pool } = require('pg');

// Load environment variables
dotenv.config();

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 8080;

// Create a PostgreSQL connection pool
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME || 'tick_data',
  user: process.env.DB_USER || 'clayb',
  password: process.env.DB_PASSWORD || 'musicman',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Simple health check endpoint
app.get('/healthcheck', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date() });
});

// API Routes
app.get('/api/bots', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM sim_bots ORDER BY bot_id');
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching bots:', error);
    res.status(500).json({ error: 'Failed to fetch bots' });
  }
});

app.get('/api/bots/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const result = await pool.query('SELECT * FROM sim_bots WHERE bot_id = $1', [id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Bot not found' });
    }
    
    // Get bot trades
    const tradesResult = await pool.query(
      'SELECT * FROM sim_bot_trades WHERE bot_id = $1 ORDER BY entry_time DESC', 
      [id]
    );
    
    // Get bot metrics
    const metricsResult = await pool.query(
      'SELECT * FROM bot_metrics WHERE bot_id = $1', 
      [id]
    );
    
    res.json({
      ...result.rows[0],
      trades: tradesResult.rows,
      metrics: metricsResult.rows[0] || null,
    });
  } catch (error) {
    console.error(`Error fetching bot ${req.params.id}:`, error);
    res.status(500).json({ error: 'Failed to fetch bot details' });
  }
});

app.get('/api/trades', async (req, res) => {
  try {
    const limit = req.query.limit ? parseInt(req.query.limit) : 100;
    
    const result = await pool.query(
      `SELECT t.*, b.name AS bot_name, b.algorithm_type
       FROM sim_bot_trades t
       JOIN sim_bots b ON t.bot_id = b.bot_id
       ORDER BY t.entry_time DESC
       LIMIT $1`,
      [limit]
    );
    
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching trades:', error);
    res.status(500).json({ error: 'Failed to fetch trades' });
  }
});

app.get('/api/trades/open', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT t.*, b.name AS bot_name, b.algorithm_type
       FROM sim_bot_trades t
       JOIN sim_bots b ON t.bot_id = b.bot_id
       WHERE t.trade_status = 'open'
       ORDER BY t.entry_time DESC`
    );
    
    res.json(result.rows);
  } catch (error) {
    console.error('Error fetching open trades:', error);
    res.status(500).json({ error: 'Failed to fetch open trades' });
  }
});

app.get('/api/metrics', async (req, res) => {
  try {
    // First try with rank_score (if column exists)
    try {
      const result = await pool.query('SELECT * FROM bot_metrics ORDER BY rank_score DESC');
      res.json(result.rows);
    } catch (rankScoreError) {
      // If rank_score column doesn't exist, try fallback
      if (rankScoreError.code === '42703' && rankScoreError.message.includes('column "rank_score" does not exist')) {
        console.log('rank_score column not found in bot_metrics table. Using total_pnl for ordering instead.');
        // Fallback to ordering by total_pnl if rank_score doesn't exist
        const result = await pool.query('SELECT * FROM bot_metrics ORDER BY total_pnl DESC');
        
        // Add a synthetic rank_score field
        const responseData = result.rows.map(row => {
          const pnl = parseFloat(row.total_pnl || 0);
          // Normalize to a 0-1 range (rough estimate)
          const syntheticRankScore = Math.min(1, Math.max(0, (pnl + 1000) / 2000));
          return {
            ...row,
            rank_score: syntheticRankScore.toFixed(4)
          };
        });
        
        return res.json(responseData);
      } else {
        // If it's some other error, rethrow
        throw rankScoreError;
      }
    }
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).json({ 
      error: 'Failed to fetch metrics',
      message: error.message
    });
  }
});

app.get('/api/dashboard', async (req, res) => {
  try {
    // Get bots and open trades
    const [botsResult, openTradesResult] = await Promise.all([
      pool.query('SELECT * FROM sim_bots'),
      pool.query(
        `SELECT t.*, b.name AS bot_name, b.algorithm_type
         FROM sim_bot_trades t
         JOIN sim_bots b ON t.bot_id = b.bot_id
         WHERE t.trade_status = 'open'
         ORDER BY t.entry_time DESC`
      )
    ]);
    
    const bots = botsResult.rows;
    const openTrades = openTradesResult.rows;
    
    // Try to get metrics with rank_score or fall back to total_pnl
    let metrics = [];
    try {
      const metricsResult = await pool.query('SELECT * FROM bot_metrics ORDER BY rank_score DESC');
      metrics = metricsResult.rows;
    } catch (rankScoreError) {
      if (rankScoreError.code === '42703' && rankScoreError.message.includes('column "rank_score" does not exist')) {
        console.log('rank_score column not found in bot_metrics table. Using total_pnl for ordering instead.');
        const metricsResult = await pool.query('SELECT * FROM bot_metrics ORDER BY total_pnl DESC');
        
        // Add a synthetic rank_score field
        metrics = metricsResult.rows.map(row => {
          const pnl = parseFloat(row.total_pnl || 0);
          const syntheticRankScore = Math.min(1, Math.max(0, (pnl + 1000) / 2000));
          return {
            ...row,
            rank_score: syntheticRankScore.toFixed(4)
          };
        });
      } else {
        throw rankScoreError;
      }
    }
    
    // Get recent trades
    const recentTradesResult = await pool.query(
      `SELECT t.*, b.name AS bot_name, b.algorithm_type
       FROM sim_bot_trades t
       JOIN sim_bots b ON t.bot_id = b.bot_id
       ORDER BY t.entry_time DESC
       LIMIT 10`
    );
    
    const recentTrades = recentTradesResult.rows;
    
    // Calculate overall system metrics
    const totalBots = bots.length;
    const activeBots = bots.filter(bot => bot.is_active).length;
    const totalOpenTrades = openTrades.length;
    
    // Calculate total P&L across all bots
    const totalPnl = metrics.reduce((sum, bot) => sum + parseFloat(bot.total_pnl || 0), 0);
    
    // Calculate average win rate
    const botsWithTrades = metrics.filter(bot => bot.total_trades > 0);
    const avgWinRate = botsWithTrades.length > 0 
      ? botsWithTrades.reduce((sum, bot) => sum + parseFloat(bot.win_rate || 0), 0) / botsWithTrades.length
      : 0;
    
    // Get top performing bots - sort by rank_score if available, else by total_pnl
    const topBots = metrics
      .filter(bot => bot.total_trades > 0)
      .sort((a, b) => {
        // Use rank_score if it exists, otherwise use total_pnl
        if (a.rank_score !== undefined && b.rank_score !== undefined) {
          return parseFloat(b.rank_score) - parseFloat(a.rank_score);
        } else {
          return parseFloat(b.total_pnl || 0) - parseFloat(a.total_pnl || 0);
        }
      })
      .slice(0, 5);
    
    res.json({
      summary: {
        totalBots,
        activeBots,
        totalOpenTrades,
        totalPnl,
        avgWinRate,
      },
      topBots,
      recentTrades,
      openTrades: openTrades.slice(0, 10),
    });
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    res.status(500).json({ 
      error: 'Failed to fetch dashboard data',
      message: error.message 
    });
  }
});

app.get('/api/allocation', async (req, res) => {
  try {
    let topBots = [];
    
    // Try with rank_score first
    try {
      const metricsResult = await pool.query(
        `SELECT m.*, b.name, b.ticker, b.algorithm_type
         FROM bot_metrics m
         JOIN sim_bots b ON m.bot_id = b.bot_id
         WHERE b.is_active = true
         ORDER BY m.rank_score DESC
         LIMIT 10`
      );
      topBots = metricsResult.rows;
    } catch (rankScoreError) {
      // If rank_score column doesn't exist, sort by total_pnl
      if (rankScoreError.code === '42703' && rankScoreError.message.includes('column "m.rank_score" does not exist')) {
        console.log('rank_score column not found in bot_metrics table. Using total_pnl for ordering instead.');
        
        // Get metrics with synthetic rank_score
        const metricsResult = await pool.query(
          `SELECT m.*, b.name, b.ticker, b.algorithm_type
           FROM bot_metrics m
           JOIN sim_bots b ON m.bot_id = b.bot_id
           WHERE b.is_active = true
           ORDER BY m.total_pnl DESC
           LIMIT 10`
        );
        
        // Add synthetic rank_score field
        topBots = metricsResult.rows.map(row => {
          const pnl = parseFloat(row.total_pnl || 0);
          const syntheticRankScore = Math.min(1, Math.max(0, (pnl + 1000) / 2000));
          return {
            ...row,
            rank_score: syntheticRankScore.toFixed(4)
          };
        });
      } else {
        throw rankScoreError;
      }
    }
    
    // Calculate allocation (for now, equal distribution among top 10)
    const allocationPerBot = 2000; // $2,000 per bot
    const allocations = topBots.map(bot => ({
      bot_id: bot.bot_id,
      name: bot.name || `Bot ${bot.bot_id}`,
      ticker: bot.ticker,
      algorithm_type: bot.algorithm_type,
      rank_score: bot.rank_score,
      allocation: allocationPerBot,
      allocation_percent: (allocationPerBot / (topBots.length * allocationPerBot)) * 100
    }));
    
    res.json({
      totalAllocation: topBots.length * allocationPerBot,
      allocations
    });
  } catch (error) {
    console.error('Error fetching allocation data:', error);
    
    // Fallback to mock data if there's a database error
    const mockAllocations = [
      { bot_id: 1, name: 'TSLA_Breakout_Bot', ticker: 'TSLA', algorithm_type: 'breakout', rank_score: 0.92, allocation: 2000, allocation_percent: 20 },
      { bot_id: 3, name: 'NVDA_Breakout_Bot', ticker: 'NVDA', algorithm_type: 'breakout', rank_score: 0.89, allocation: 2000, allocation_percent: 20 },
      { bot_id: 5, name: 'AAPL_Support_Resistance_Bot', ticker: 'AAPL', algorithm_type: 'support_resistance', rank_score: 0.87, allocation: 2000, allocation_percent: 20 },
      { bot_id: 2, name: 'COIN_Momentum_Bot', ticker: 'COIN', algorithm_type: 'momentum', rank_score: 0.78, allocation: 2000, allocation_percent: 20 },
      { bot_id: 4, name: 'AMD_Momentum_Bot', ticker: 'AMD', algorithm_type: 'momentum', rank_score: 0.75, allocation: 2000, allocation_percent: 20 }
    ];
    
    console.log('Falling back to mock allocation data');
    res.json({
      totalAllocation: mockAllocations.length * 2000,
      allocations: mockAllocations
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});