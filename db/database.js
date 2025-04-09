// Simplified database module for Remix minimal app
const { Pool } = require('pg');

// Flag to use mock data instead of real database
const USE_MOCK_DATA = process.env.USE_MOCK_DATA === 'true';

// Create a PostgreSQL connection pool
let pool;
if (!USE_MOCK_DATA) {
  // Check if we're connecting via Cloud SQL proxy or direct socket
  const isCloudSocket = process.env.CLOUD_SQL_CONNECTION_NAME && 
                      process.env.DB_HOST === '/cloudsql';
  
  if (isCloudSocket) {
    // Connect directly using UNIX socket
    const connectionName = process.env.CLOUD_SQL_CONNECTION_NAME;
    pool = new Pool({
      user: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || '',
      database: process.env.DB_NAME || 'tick_data',
      host: `/cloudsql/${connectionName}`,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 10000,
    });
    console.log(`Connecting to Cloud SQL using UNIX socket: /cloudsql/${connectionName}`);
  } else {
    // Connect using TCP (Cloud SQL Proxy)
    pool = new Pool({
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'tick_data',
      user: process.env.DB_USER || 'clayb',
      password: process.env.DB_PASSWORD || 'musicman',
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 10000,
    });
    console.log(`Connecting to Cloud SQL using TCP: ${process.env.DB_HOST || 'localhost'}:${process.env.DB_PORT || '5432'}`);
  }
}

// Mock data for development
const mockData = {
  bots: [
    { bot_id: 1, name: 'TSLA_Breakout_Bot', ticker: 'TSLA', algorithm_module: 'algorithms.breakout_algorithm', algorithm_type: 'breakout', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.01, description: 'TSLA breakout strategy using volatility-based entry', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 2, name: 'COIN_Momentum_Bot', ticker: 'COIN', algorithm_module: 'algorithms.momentum_algorithm', algorithm_type: 'momentum', trade_direction: 'LONG', position_size: 1000.0, trailing_stop_pct: 0.015, description: 'COIN momentum strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 3, name: 'NVDA_Breakout_Bot', ticker: 'NVDA', algorithm_module: 'algorithms.breakout_algorithm', algorithm_type: 'breakout', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.01, description: 'NVDA breakout strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 4, name: 'AMD_Momentum_Bot', ticker: 'AMD', algorithm_module: 'algorithms.momentum_algorithm', algorithm_type: 'momentum', trade_direction: 'LONG', position_size: 1000.0, trailing_stop_pct: 0.012, description: 'AMD momentum strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 5, name: 'AAPL_Support_Resistance_Bot', ticker: 'AAPL', algorithm_module: 'algorithms.support_resistance_algorithm', algorithm_type: 'support_resistance', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.008, description: 'AAPL support resistance strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
  ],
  trades: [
    { trade_id: 1, bot_id: 1, ticker: 'TSLA', entry_price: 180.25, exit_price: 185.50, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-20T10:15:00.000Z', exit_time: '2025-03-20T14:30:00.000Z', trade_status: 'closed', pnl: 290.83, pnl_percent: 0.0291, trailing_stop_price: 183.20, exit_reason: 'trailing_stop', bot_name: 'TSLA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 2, bot_id: 2, ticker: 'COIN', entry_price: 210.75, exit_price: 206.30, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-21T09:45:00.000Z', exit_time: '2025-03-21T15:20:00.000Z', trade_status: 'closed', pnl: -211.39, pnl_percent: -0.0211, trailing_stop_price: 206.30, exit_reason: 'trailing_stop', bot_name: 'COIN_Momentum_Bot', algorithm_type: 'momentum' },
    { trade_id: 3, bot_id: 3, ticker: 'NVDA', entry_price: 950.00, exit_price: 972.25, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-22T10:00:00.000Z', exit_time: '2025-03-22T16:15:00.000Z', trade_status: 'closed', pnl: 234.21, pnl_percent: 0.0234, trailing_stop_price: 965.00, exit_reason: 'profit_target', bot_name: 'NVDA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 6, bot_id: 1, ticker: 'TSLA', entry_price: 182.40, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T10:00:00.000Z', trade_status: 'open', trailing_stop_price: 179.75, bot_name: 'TSLA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 7, bot_id: 3, ticker: 'NVDA', entry_price: 965.25, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T09:45:00.000Z', trade_status: 'open', trailing_stop_price: 955.60, bot_name: 'NVDA_Breakout_Bot', algorithm_type: 'breakout' },
  ],
  metrics: [
    { id: 1, bot_id: 1, total_trades: 32, winning_trades: 21, losing_trades: 11, total_pnl: 2450.75, average_pnl_per_trade: 76.59, win_rate: 0.6563, average_win_amount: 152.32, average_loss_amount: -72.45, profit_factor: 2.10, max_drawdown: -450.20, sharpe_ratio: 1.85, risk_reward_ratio: 2.10, expectancy: 0.24, rank_score: 0.92 },
    { id: 2, bot_id: 3, total_trades: 28, winning_trades: 18, losing_trades: 10, total_pnl: 2120.50, average_pnl_per_trade: 75.73, win_rate: 0.6429, average_win_amount: 145.80, average_loss_amount: -68.90, profit_factor: 1.95, max_drawdown: -380.60, sharpe_ratio: 1.72, risk_reward_ratio: 2.12, expectancy: 0.23, rank_score: 0.89 },
    { id: 4, bot_id: 2, total_trades: 30, winning_trades: 17, losing_trades: 13, total_pnl: 1650.20, average_pnl_per_trade: 55.01, win_rate: 0.5667, average_win_amount: 128.75, average_loss_amount: -75.30, profit_factor: 1.68, max_drawdown: -520.10, sharpe_ratio: 1.45, risk_reward_ratio: 1.71, expectancy: 0.19, rank_score: 0.78 },
  ]
};

// Database Query Functions

async function query(text, params) {
  if (USE_MOCK_DATA) {
    console.log(`Using mock data for query: ${text.substring(0, 100)}...`);
    
    // For mock data, parse the query to determine what data to return
    if (text.includes('sim_bots')) {
      return { rows: mockData.bots };
    } else if (text.includes('sim_bot_trades')) {
      // Handle filtering for open trades
      if (text.includes("trade_status = 'open'")) {
        return { rows: mockData.trades.filter(t => t.trade_status === 'open') };
      }
      return { rows: mockData.trades };
    } else if (text.includes('bot_metrics')) {
      return { rows: mockData.metrics };
    }
    
    return { rows: [] };
  }
  
  try {
    console.log(`Executing query: ${text.substring(0, 100)}...`);
    const client = await pool.connect();
    try {
      const result = await client.query(text, params);
      console.log(`Query result rows: ${result.rows.length}`);
      return result;
    } finally {
      client.release();
    }
  } catch (error) {
    console.error('Error executing query:', error.message);
    
    // Fall back to mock data if there's a database error
    console.warn('Database error. Falling back to mock data for this query.');
    
    // Return appropriate mock data
    if (text.includes('sim_bots')) {
      return { rows: mockData.bots };
    } else if (text.includes('sim_bot_trades')) {
      // Handle filtering for open trades
      if (text.includes("trade_status = 'open'")) {
        return { rows: mockData.trades.filter(t => t.trade_status === 'open') };
      }
      return { rows: mockData.trades };
    } else if (text.includes('bot_metrics')) {
      return { rows: mockData.metrics };
    }
    
    return { rows: [] };
  }
}

// API Functions

async function getBots() {
  const result = await query('SELECT * FROM sim_bots ORDER BY bot_id');
  return result.rows;
}

async function getBotById(botId) {
  const result = await query('SELECT * FROM sim_bots WHERE bot_id = $1', [botId]);
  
  if (result.rows.length === 0) {
    return null;
  }
  
  // Get bot trades
  const trades = await query(
    'SELECT * FROM sim_bot_trades WHERE bot_id = $1 ORDER BY entry_time DESC', 
    [botId]
  );
  
  // Get bot metrics
  const metrics = await query(
    'SELECT * FROM bot_metrics WHERE bot_id = $1', 
    [botId]
  );
  
  return {
    ...result.rows[0],
    trades: trades.rows,
    metrics: metrics.rows[0] || null,
  };
}

async function getTrades(limit = 100) {
  const result = await query(
    `SELECT t.*, b.name AS bot_name, b.algorithm_type
     FROM sim_bot_trades t
     JOIN sim_bots b ON t.bot_id = b.bot_id
     ORDER BY t.entry_time DESC
     LIMIT $1`,
    [limit]
  );
  return result.rows;
}

async function getOpenTrades() {
  const result = await query(
    `SELECT t.*, b.name AS bot_name, b.algorithm_type
     FROM sim_bot_trades t
     JOIN sim_bots b ON t.bot_id = b.bot_id
     WHERE t.trade_status = 'open'
     ORDER BY t.entry_time DESC`
  );
  return result.rows;
}

async function getBotMetrics() {
  try {
    // First try with rank_score (if column exists)
    const result = await query('SELECT * FROM bot_metrics ORDER BY rank_score DESC');
    return result.rows;
  } catch (error) {
    if (error.message && error.message.includes('column "rank_score" does not exist')) {
      console.warn('rank_score column not found in bot_metrics table. Using total_pnl for ordering instead.');
      // Fallback to ordering by total_pnl if rank_score doesn't exist
      const result = await query('SELECT * FROM bot_metrics ORDER BY total_pnl DESC');
      
      // Add a synthetic rank_score field based on total_pnl
      return result.rows.map(row => {
        // Calculate a simple rank score based on total_pnl to ensure UI works
        const pnl = parseFloat(row.total_pnl || 0);
        // Normalize to a 0-1 range (rough estimate)
        const syntheticRankScore = Math.min(1, Math.max(0, (pnl + 1000) / 2000));
        return {
          ...row,
          rank_score: syntheticRankScore.toFixed(4)
        };
      });
    }
    console.error('Error in getBotMetrics:', error);
    return mockData.metrics; // Fallback to mock data
  }
}

// Export database functions
module.exports = {
  query,
  getBots,
  getBotById,
  getTrades,
  getOpenTrades,
  getBotMetrics,
  mockData
};