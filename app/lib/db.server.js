import pkg from 'pg';
import { getEnv } from './env.server';
const { Pool } = pkg;

// Get environment configuration
const env = getEnv();

// Flag to use mock data instead of real database
const USE_MOCK_DATA = env.USE_MOCK_DATA || false;

// Create a PostgreSQL connection pool (only if not using mock data)
let pool;
if (!USE_MOCK_DATA) {
  pool = new Pool({
    host: env.DB_HOST,
    port: env.DB_PORT,
    database: env.DB_NAME,
    user: env.DB_USER,
    password: env.DB_PASSWORD,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000, // Extended timeout for initial connection
  });
}

// Mock data for development
const mockData = {
  bots: [
    { bot_id: 1, name: 'TSLA_Breakout_Bot', ticker: 'TSLA', algorithm_module: 'algorithms.breakout_algorithm', algorithm_type: 'breakout', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.01, description: 'TSLA breakout strategy using volatility-based entry', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 2, name: 'COIN_Momentum_Bot', ticker: 'COIN', algorithm_module: 'algorithms.momentum_algorithm', algorithm_type: 'momentum', trade_direction: 'LONG', position_size: 1000.0, trailing_stop_pct: 0.015, description: 'COIN momentum strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 3, name: 'NVDA_Breakout_Bot', ticker: 'NVDA', algorithm_module: 'algorithms.breakout_algorithm', algorithm_type: 'breakout', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.01, description: 'NVDA breakout strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 4, name: 'AMD_Momentum_Bot', ticker: 'AMD', algorithm_module: 'algorithms.momentum_algorithm', algorithm_type: 'momentum', trade_direction: 'LONG', position_size: 1000.0, trailing_stop_pct: 0.012, description: 'AMD momentum strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 5, name: 'AAPL_Support_Resistance_Bot', ticker: 'AAPL', algorithm_module: 'algorithms.support_resistance_algorithm', algorithm_type: 'support_resistance', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.008, description: 'AAPL support resistance strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 6, name: 'MSFT_Mean_Reversion_Bot', ticker: 'MSFT', algorithm_module: 'algorithms.mean_reversion_algorithm', algorithm_type: 'mean_reversion', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.009, description: 'MSFT mean reversion strategy', version: '1.0', is_active: false, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 7, name: 'META_Volatility_Bot', ticker: 'META', algorithm_module: 'algorithms.volatility_breakout_algorithm', algorithm_type: 'volatility_breakout', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.011, description: 'META volatility breakout strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
    { bot_id: 8, name: 'AMZN_Price_Pattern_Bot', ticker: 'AMZN', algorithm_module: 'algorithms.price_pattern_algorithm', algorithm_type: 'price_pattern', trade_direction: 'BOTH', position_size: 1000.0, trailing_stop_pct: 0.01, description: 'AMZN price pattern strategy', version: '1.0', is_active: true, created_at: '2025-03-01T00:00:00.000Z', last_updated: '2025-03-01T00:00:00.000Z' },
  ],
  trades: [
    { trade_id: 1, bot_id: 1, ticker: 'TSLA', entry_price: 180.25, exit_price: 185.50, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-20T10:15:00.000Z', exit_time: '2025-03-20T14:30:00.000Z', trade_status: 'closed', pnl: 290.83, pnl_percent: 0.0291, trailing_stop_price: 183.20, exit_reason: 'trailing_stop', bot_name: 'TSLA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 2, bot_id: 2, ticker: 'COIN', entry_price: 210.75, exit_price: 206.30, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-21T09:45:00.000Z', exit_time: '2025-03-21T15:20:00.000Z', trade_status: 'closed', pnl: -211.39, pnl_percent: -0.0211, trailing_stop_price: 206.30, exit_reason: 'trailing_stop', bot_name: 'COIN_Momentum_Bot', algorithm_type: 'momentum' },
    { trade_id: 3, bot_id: 3, ticker: 'NVDA', entry_price: 950.00, exit_price: 972.25, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-22T10:00:00.000Z', exit_time: '2025-03-22T16:15:00.000Z', trade_status: 'closed', pnl: 234.21, pnl_percent: 0.0234, trailing_stop_price: 965.00, exit_reason: 'profit_target', bot_name: 'NVDA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 4, bot_id: 4, ticker: 'AMD', entry_price: 172.50, exit_price: 175.20, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-23T09:30:00.000Z', exit_time: '2025-03-23T14:45:00.000Z', trade_status: 'closed', pnl: 156.52, pnl_percent: 0.0157, trailing_stop_price: 173.80, exit_reason: 'profit_target', bot_name: 'AMD_Momentum_Bot', algorithm_type: 'momentum' },
    { trade_id: 5, bot_id: 5, ticker: 'AAPL', entry_price: 185.30, exit_price: 182.75, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-24T11:15:00.000Z', exit_time: '2025-03-24T15:30:00.000Z', trade_status: 'closed', pnl: -137.61, pnl_percent: -0.0138, trailing_stop_price: 182.75, exit_reason: 'trailing_stop', bot_name: 'AAPL_Support_Resistance_Bot', algorithm_type: 'support_resistance' },
    { trade_id: 6, bot_id: 1, ticker: 'TSLA', entry_price: 182.40, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T10:00:00.000Z', trade_status: 'open', trailing_stop_price: 179.75, bot_name: 'TSLA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 7, bot_id: 3, ticker: 'NVDA', entry_price: 965.25, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T09:45:00.000Z', trade_status: 'open', trailing_stop_price: 955.60, bot_name: 'NVDA_Breakout_Bot', algorithm_type: 'breakout' },
    { trade_id: 8, bot_id: 7, ticker: 'META', entry_price: 485.00, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T10:30:00.000Z', trade_status: 'open', trailing_stop_price: 480.15, bot_name: 'META_Volatility_Bot', algorithm_type: 'volatility_breakout' },
    { trade_id: 9, bot_id: 8, ticker: 'AMZN', entry_price: 180.50, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T11:00:00.000Z', trade_status: 'open', trailing_stop_price: 178.70, bot_name: 'AMZN_Price_Pattern_Bot', algorithm_type: 'price_pattern' },
    { trade_id: 10, bot_id: 2, ticker: 'COIN', entry_price: 208.25, trade_size: 1000, trade_direction: 'LONG', entry_time: '2025-03-25T10:15:00.000Z', trade_status: 'open', trailing_stop_price: 205.10, bot_name: 'COIN_Momentum_Bot', algorithm_type: 'momentum' },
  ],
  metrics: [
    { id: 1, bot_id: 1, total_trades: 32, winning_trades: 21, losing_trades: 11, total_pnl: 2450.75, average_pnl_per_trade: 76.59, win_rate: 0.6563, average_win_amount: 152.32, average_loss_amount: -72.45, profit_factor: 2.10, max_drawdown: -450.20, sharpe_ratio: 1.85, risk_reward_ratio: 2.10, expectancy: 0.24, rank_score: 0.92 },
    { id: 2, bot_id: 3, total_trades: 28, winning_trades: 18, losing_trades: 10, total_pnl: 2120.50, average_pnl_per_trade: 75.73, win_rate: 0.6429, average_win_amount: 145.80, average_loss_amount: -68.90, profit_factor: 1.95, max_drawdown: -380.60, sharpe_ratio: 1.72, risk_reward_ratio: 2.12, expectancy: 0.23, rank_score: 0.89 },
    { id: 3, bot_id: 7, total_trades: 25, winning_trades: 16, losing_trades: 9, total_pnl: 1870.30, average_pnl_per_trade: 74.81, win_rate: 0.6400, average_win_amount: 142.50, average_loss_amount: -65.80, profit_factor: 1.92, max_drawdown: -350.40, sharpe_ratio: 1.68, risk_reward_ratio: 2.17, expectancy: 0.22, rank_score: 0.87 },
    { id: 4, bot_id: 2, total_trades: 30, winning_trades: 17, losing_trades: 13, total_pnl: 1650.20, average_pnl_per_trade: 55.01, win_rate: 0.5667, average_win_amount: 128.75, average_loss_amount: -75.30, profit_factor: 1.68, max_drawdown: -520.10, sharpe_ratio: 1.45, risk_reward_ratio: 1.71, expectancy: 0.19, rank_score: 0.78 },
    { id: 5, bot_id: 8, total_trades: 22, winning_trades: 12, losing_trades: 10, total_pnl: 1320.60, average_pnl_per_trade: 60.03, win_rate: 0.5455, average_win_amount: 135.40, average_loss_amount: -72.10, profit_factor: 1.62, max_drawdown: -410.50, sharpe_ratio: 1.38, risk_reward_ratio: 1.88, expectancy: 0.18, rank_score: 0.75 },
    { id: 6, bot_id: 4, total_trades: 24, winning_trades: 14, losing_trades: 10, total_pnl: 1410.80, average_pnl_per_trade: 58.78, win_rate: 0.5833, average_win_amount: 125.60, average_loss_amount: -69.80, profit_factor: 1.65, max_drawdown: -390.30, sharpe_ratio: 1.42, risk_reward_ratio: 1.80, expectancy: 0.17, rank_score: 0.74 },
    { id: 7, bot_id: 5, total_trades: 26, winning_trades: 13, losing_trades: 13, total_pnl: 980.25, average_pnl_per_trade: 37.70, win_rate: 0.5000, average_win_amount: 118.90, average_loss_amount: -74.20, profit_factor: 1.42, max_drawdown: -480.70, sharpe_ratio: 1.20, risk_reward_ratio: 1.60, expectancy: 0.15, rank_score: 0.68 },
    { id: 8, bot_id: 6, total_trades: 20, winning_trades: 11, losing_trades: 9, total_pnl: 750.40, average_pnl_per_trade: 37.52, win_rate: 0.5500, average_win_amount: 110.80, average_loss_amount: -70.60, profit_factor: 1.35, max_drawdown: -420.90, sharpe_ratio: 1.15, risk_reward_ratio: 1.57, expectancy: 0.14, rank_score: 0.65 },
  ],
};

// Generate additional bots from 9-126
for (let i = 9; i <= 126; i++) {
  const strategies = ['breakout', 'momentum', 'mean_reversion', 'support_resistance', 'volatility_breakout', 'price_pattern'];
  const tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'GOOGL', 'META', 'COIN', 'AMD', 'NFLX', 'IBM', 'INTC'];
  const directions = ['LONG', 'SHORT', 'BOTH'];
  
  const strategyIndex = i % strategies.length;
  const tickerIndex = i % tickers.length;
  const directionIndex = i % directions.length;
  
  const strategy = strategies[strategyIndex];
  const ticker = tickers[tickerIndex];
  const direction = directions[directionIndex];
  
  mockData.bots.push({
    bot_id: i,
    name: `${ticker}_${strategy}_Bot_${i}`,
    ticker: ticker,
    algorithm_module: `algorithms.${strategy.replace('_', '_')}_algorithm`,
    algorithm_type: strategy,
    trade_direction: direction,
    position_size: 1000.0 + (i % 5) * 500,
    trailing_stop_pct: 0.005 + (i % 10) * 0.001,
    description: `${ticker} ${strategy.replace('_', ' ')} strategy #${i}`,
    version: '1.0',
    is_active: i % 8 !== 0, // Make every 8th bot inactive
    created_at: '2025-03-01T00:00:00.000Z',
    last_updated: '2025-03-01T00:00:00.000Z',
  });
}

// Add metrics for all bots
for (let i = 1; i <= 126; i++) {
  if (!mockData.metrics.find(m => m.bot_id === i)) {
    // Base values that get slightly randomized
    const winRate = 0.45 + (Math.random() * 0.3);
    const profitFactor = 1.0 + (Math.random() * 1.5);
    const totalPnl = (500 + Math.random() * 2500) * (Math.random() > 0.2 ? 1 : -1); // 20% chance of negative PnL
    
    mockData.metrics.push({
      id: mockData.metrics.length + 1,
      bot_id: i,
      total_trades: 10 + Math.floor(Math.random() * 30),
      winning_trades: Math.floor(winRate * (10 + Math.floor(Math.random() * 30))),
      losing_trades: Math.floor((1 - winRate) * (10 + Math.floor(Math.random() * 30))),
      total_pnl: totalPnl,
      average_pnl_per_trade: totalPnl / (10 + Math.floor(Math.random() * 30)),
      win_rate: winRate,
      average_win_amount: 80 + Math.random() * 100,
      average_loss_amount: -(30 + Math.random() * 70),
      profit_factor: profitFactor,
      max_drawdown: -(100 + Math.random() * 500),
      sharpe_ratio: 0.8 + Math.random() * 1.5,
      risk_reward_ratio: 1.0 + Math.random() * 1.5,
      expectancy: 0.1 + Math.random() * 0.3,
      rank_score: 0.4 + Math.random() * 0.6
    });
  }
}

// Generate bot data with parameters for testing
mockData.bots = mockData.bots.map(bot => {
  bot.parameters = {
    lookback_period: Math.floor(Math.random() * 10) + 15,
    volatility_threshold: (Math.random() * 1.5 + 1).toFixed(2),
    profit_target_pct: (Math.random() * 0.03 + 0.01).toFixed(3),
    stop_loss_pct: (Math.random() * 0.01 + 0.005).toFixed(3),
    rsi_upper: Math.floor(Math.random() * 10) + 65,
    rsi_lower: Math.floor(Math.random() * 10) + 25,
    moving_average_period: Math.floor(Math.random() * 10) + 10,
  };
  return bot;
});

export async function getConnection() {
  if (USE_MOCK_DATA) {
    return {
      query: () => Promise.resolve({ rows: [] }),
      release: () => {},
    };
  }
  
  try {
    console.log("Attempting to connect to PostgreSQL database...");
    // Try to connect directly to the main pool
    return await pool.connect();
  } catch (error) {
    console.error('Error connecting to database:', error.message);
    console.log("Falling back to mock data due to connection error");
    // Switch to mock data mode
    Object.defineProperty(exports, 'USE_MOCK_DATA', { value: true });
    return {
      query: () => Promise.resolve({ rows: [] }),
      release: () => {},
    };
  }
}

export async function query(text, params) {
  if (USE_MOCK_DATA) {
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
    console.log(`Using mock data for query: ${text.substring(0, 100)}...`);
    return { rows: [] };
  }
  
  const client = await getConnection();
  try {
    console.log(`Executing query: ${text.substring(0, 100)}...`);
    const result = await client.query(text, params);
    console.log(`Query result rows: ${result.rows.length}`);
    return result;
  } catch (error) {
    console.error('Error executing query:', error.message);
    
    // If we get specific DB errors, fall back to mock data
    if (error.code === '3D000' || error.code === '42P01' || error.code === '28P01' || 
        error.code === 'ECONNREFUSED' || error.code === '08006' || error.code === '57P03') {
      console.warn('Database error. Falling back to mock data for this query.');
      // Switch to mock data mode globally
      Object.defineProperty(exports, 'USE_MOCK_DATA', { value: true });
      
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
    }
    throw error;
  } finally {
    client.release();
  }
}

export async function getBots() {
  if (USE_MOCK_DATA) {
    return mockData.bots;
  }
  
  const result = await query('SELECT * FROM sim_bots ORDER BY bot_id');
  return result.rows;
}

export async function getTrades(limit = 100) {
  if (USE_MOCK_DATA) {
    return mockData.trades.slice(0, limit);
  }
  
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

export async function getOpenTrades() {
  if (USE_MOCK_DATA) {
    return mockData.trades.filter(trade => trade.trade_status === 'open');
  }
  
  const result = await query(
    `SELECT t.*, b.name AS bot_name, b.algorithm_type
     FROM sim_bot_trades t
     JOIN sim_bots b ON t.bot_id = b.bot_id
     WHERE t.trade_status = 'open'
     ORDER BY t.entry_time DESC`
  );
  return result.rows;
}

export async function getBotMetrics() {
  if (USE_MOCK_DATA) {
    console.log('Using mock data for bot metrics');
    return mockData.metrics;
  }
  
  try {
    // First try with rank_score (if column exists)
    console.log('Querying real database for bot metrics');
    const result = await query('SELECT * FROM bot_metrics ORDER BY rank_score DESC');
    
    // Make sure data is properly formatted
    return result.rows.map(row => {
      return {
        ...row,
        win_rate: parseFloat(row.win_rate || 0).toString(),
        profit_factor: parseFloat(row.profit_factor || 0).toString(),
        total_pnl: parseFloat(row.total_pnl || 0).toString(),
        average_win_amount: parseFloat(row.average_win_amount || 0).toString(),
        average_loss_amount: parseFloat(row.average_loss_amount || 0).toString(),
        max_drawdown: parseFloat(row.max_drawdown || 0).toString(),
        sharpe_ratio: parseFloat(row.sharpe_ratio || 0).toString(),
        risk_reward_ratio: parseFloat(row.risk_reward_ratio || 0).toString(),
        expectancy: parseFloat(row.expectancy || 0).toString(),
        rank_score: parseFloat(row.rank_score || 0).toString(),
      };
    });
  } catch (error) {
    console.warn('Error fetching bot metrics:', error.message);
    console.log('Falling back to mock data for bot metrics');
    return mockData.metrics;
  }
}

export async function getBotById(botId) {
  if (USE_MOCK_DATA) {
    const bot = mockData.bots.find(b => b.bot_id === botId);
    if (!bot) return null;
    
    // Get bot trades
    const trades = mockData.trades.filter(t => t.bot_id === botId);
    
    // Get bot metrics
    const metrics = mockData.metrics.find(m => m.bot_id === botId);
    
    return {
      ...bot,
      trades,
      metrics,
    };
  }
  
  // Real database implementation
  try {
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
  } catch (error) {
    console.error(`Error fetching bot ${botId}:`, error);
    throw error;
  }
}

// Export the mockData for fallback in api.server.js
export { mockData };

export default {
  getConnection,
  query,
  getBots,
  getTrades,
  getOpenTrades,
  getBotMetrics,
  getBotById,
  mockData,  // Include mockData in the default export
};