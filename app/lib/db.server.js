import pkg from 'pg';
import { getEnv } from './env.server';
const { Pool } = pkg;

// Get environment configuration
const env = getEnv();

// Always use real database
const USE_MOCK_DATA = false;

// Create a PostgreSQL connection pool with optimized settings
const pool = new Pool({
  host: env.DB_HOST,
  port: env.DB_PORT,
  database: env.DB_NAME,
  user: env.DB_USER,
  password: env.DB_PASSWORD,
  max: 10,  // Reduce max connections to avoid overwhelming the database
  idleTimeoutMillis: 10000, // Reduce idle timeout
  connectionTimeoutMillis: 5000, // Reduce connection timeout
  statement_timeout: 5000, // Set statement timeout to 5 seconds
  query_timeout: 5000,     // Set query timeout to 5 seconds
});

// Add event handlers for pool errors
pool.on('error', (err) => {
  console.error('Unexpected database pool error:', err);
});

// Track active connections for debugging
let activeConnections = 0;
pool.on('connect', () => {
  activeConnections++;
  console.log(`DB connection established. Active connections: ${activeConnections}`);
});

pool.on('remove', () => {
  activeConnections--;
  console.log(`DB connection released. Active connections: ${activeConnections}`);
});

export async function getConnection() {
  try {
    console.log("Attempting to connect to PostgreSQL database...");
    return await pool.connect();
  } catch (error) {
    console.error('Error connecting to database:', error.message);
    throw error; // Don't fall back to mock data, throw the error
  }
}

export async function query(text, params) {
  // Use a timeout to prevent long-running queries
  const QUERY_TIMEOUT = 5000;
  
  const queryPromise = new Promise(async (resolve, reject) => {
    let isTimedOut = false;
    const timeout = setTimeout(() => {
      isTimedOut = true;
      reject(new Error(`Query timed out after ${QUERY_TIMEOUT}ms: ${text.substring(0, 100)}...`));
    }, QUERY_TIMEOUT);
    
    let client;
    try {
      client = await pool.connect();
      const startTime = Date.now();
      
      // Execute the query
      const result = await client.query(text, params);
      
      // Clear timeout since query completed
      clearTimeout(timeout);
      if (isTimedOut) return; // Already rejected
      
      const duration = Date.now() - startTime;
      console.log(`Query completed in ${duration}ms`);
      
      // Log slow queries for optimization
      if (duration > 200) {
        console.warn(`SLOW QUERY (${duration}ms): ${text.substring(0, 100)}...`);
      }
      
      resolve(result);
    } catch (error) {
      // Clear timeout since query errored
      clearTimeout(timeout);
      if (isTimedOut) return; // Already rejected
      
      console.error('Error executing query:', error.message);
      reject(error);
    } finally {
      if (client) {
        client.release();
      }
    }
  });

  return queryPromise;
}

export async function getBots(limit = 500) {
  // Add limit to avoid loading too many bots, but make it high enough to get all bots
  const result = await query('SELECT * FROM sim_bots ORDER BY bot_id LIMIT $1', [limit]);
  
  // Use more efficient property mapping
  return result.rows.map(bot => ({
    bot_id: bot.bot_id,
    name: bot.name,
    ticker: bot.ticker,
    algorithm_module: bot.algorithm_module,
    algorithm_type: bot.algorithm_type,
    trade_direction: bot.trade_direction,
    position_size: Number(bot.position_size || 0),
    trailing_stop_pct: Number(bot.trailing_stop_pct || 0),
    description: bot.description || '',
    version: bot.version || '1.0',
    is_active: Boolean(bot.is_active),
    created_at: bot.created_at,
    last_updated: bot.last_updated
  }));
}

export async function getTrades(limit = 50) {
  // Use a more efficient query with only needed columns
  const result = await query(
    `SELECT 
       t.trade_id, t.bot_id, t.ticker, t.entry_price, t.exit_price, 
       t.trade_size, t.trade_direction, t.entry_time, t.exit_time, 
       t.trade_status, t.trade_pnl, t.pnl_percent, t.exit_trigger_price,
       b.name AS bot_name, b.algorithm_type
     FROM sim_bot_trades t
     JOIN sim_bots b ON t.bot_id = b.bot_id
     ORDER BY t.entry_time DESC
     LIMIT $1`,
    [limit]
  );
  
  // Map only essential properties with more efficient conversions
  return result.rows.map(row => ({
    trade_id: row.trade_id,
    bot_id: row.bot_id,
    ticker: row.ticker,
    entry_price: Number(row.entry_price || 0),
    exit_price: Number(row.exit_price || 0),
    trade_size: Number(row.trade_size || 0),
    trade_direction: row.trade_direction,
    entry_time: row.entry_time,
    exit_time: row.exit_time,
    trade_status: row.trade_status,
    pnl: Number(row.trade_pnl || 0),
    pnl_percent: Number(row.pnl_percent || 0),
    trailing_stop_price: row.exit_trigger_price,
    exit_reason: row.exit_trigger_price ? 'trigger_price' : 'unknown',
    bot_name: row.bot_name,
    algorithm_type: row.algorithm_type
  }));
}

export async function getOpenTrades(limit = 50) {
  // Use a more efficient query with only needed columns and filter in SQL
  const result = await query(
    `SELECT 
       t.trade_id, t.bot_id, t.ticker, t.entry_price, t.exit_price, 
       t.trade_size, t.trade_direction, t.entry_time, t.exit_time, 
       t.trade_status, t.trade_pnl, t.pnl_percent, t.exit_trigger_price,
       b.name AS bot_name, b.algorithm_type
     FROM sim_bot_trades t
     JOIN sim_bots b ON t.bot_id = b.bot_id
     WHERE t.trade_status = 'open'
     ORDER BY t.entry_time DESC
     LIMIT $1`,
    [limit]
  );
  
  // Map only essential properties with more efficient conversions
  return result.rows.map(row => ({
    trade_id: row.trade_id,
    bot_id: row.bot_id,
    ticker: row.ticker,
    entry_price: Number(row.entry_price || 0),
    exit_price: Number(row.exit_price || 0),
    trade_size: Number(row.trade_size || 0),
    trade_direction: row.trade_direction,
    entry_time: row.entry_time,
    exit_time: row.exit_time,
    trade_status: row.trade_status,
    pnl: Number(row.trade_pnl || 0),
    pnl_percent: Number(row.pnl_percent || 0),
    trailing_stop_price: row.exit_trigger_price,
    bot_name: row.bot_name,
    algorithm_type: row.algorithm_type
  }));
}

export async function getBotMetrics(limit = 500) {
  try {
    console.log('Querying real database for bot metrics');
    
    // First, check if we can get any data at all from the table
    const countResult = await query('SELECT COUNT(DISTINCT bot_id) FROM bot_metrics');
    console.log(`Found ${countResult.rows[0].count} unique bot_ids in bot_metrics table`);
    
    if (parseInt(countResult.rows[0].count) === 0) {
      console.log('No records found in bot_metrics table');
      return [];
    }
    
    // Get only the most recent record for each bot_id to avoid duplicates
    const result = await query(`
      SELECT m.*, b.position_size 
      FROM (
        SELECT DISTINCT ON (bot_id) * 
        FROM bot_metrics 
        ORDER BY bot_id, last_updated DESC
      ) m
      LEFT JOIN sim_bots b ON m.bot_id = b.bot_id
      LIMIT $1
    `, [limit]);
    
    console.log(`Retrieved ${result.rows.length} unique bot metrics records`);
    
    if (result.rows.length === 0) {
      console.log('Query returned zero rows');
      return [];
    }
    
    // Log the first row to see its structure
    console.log('Sample row structure:', JSON.stringify(result.rows[0]));
    
    // Map the results, handling possible missing columns
    return result.rows.map(row => {
      // Use appropriate column names based on your schema
      // Check if properties exist before using them
      
      // Handle win rate calculation from various possible column names
      let winRate = 0;
      if (row.win_rate !== undefined) {
        winRate = Number(row.win_rate || 0);
      } else if (row.avg_win_rate !== undefined) {
        winRate = Number(row.avg_win_rate || 0) / 100; // Convert from percentage
      } else if (row.winning_trades !== undefined && row.total_trades !== undefined) {
        winRate = row.total_trades > 0 ? Number(row.winning_trades) / Number(row.total_trades) : 0;
      }
      
      // Calculate rank score based on available metrics
      const profitFactor = Number(row.profit_factor || 0);
      const sharpeRatio = Number(row.sharpe_ratio || 0);
      
      const calculatedRankScore = (
        (winRate * 0.5) + 
        (profitFactor * 0.3 / 3) + 
        (sharpeRatio * 0.2 / 3)
      ).toFixed(2);
      
      // Return a standardized object, using fallbacks for missing properties
      return {
        bot_id: row.bot_id || 0,
        algo_id: row.algo_id || row.algorithm_id || 0,
        ticker: row.ticker || '',
        win_rate: winRate.toFixed(4),
        profit_factor: Number(row.profit_factor || 0).toFixed(2),
        total_pnl: Number(row.total_pnl || 0).toFixed(2),
        average_win_amount: Number(row.avg_profit_per_trade || row.average_win_amount || 0).toFixed(2),
        average_loss_amount: (-Math.abs(Number(row.avg_drawdown || row.average_loss_amount || 0))).toFixed(2),
        max_drawdown: (-Math.abs(Number(row.max_drawdown || 0))).toFixed(2),
        sharpe_ratio: Number(row.sharpe_ratio || 0).toFixed(2),
        risk_reward_ratio: Number(row.r_multiple || row.risk_reward_ratio || 0).toFixed(2),
        total_trades: Number(row.total_trades || 0),
        winning_trades: Number(row.winning_trades || Math.round(Number(row.total_trades || 0) * winRate)),
        losing_trades: Number(row.losing_trades || Math.round(Number(row.total_trades || 0) * (1 - winRate))),
        rank_score: calculatedRankScore,
        last_updated: row.last_updated,
        drawdown_percent: Number(row.drawdown_percent || 0).toFixed(2),
        position_size: Number(row.position_size || 10000).toFixed(2),
        pnl_percent: Number(row.one_month_performance || row.one_week_performance || 0).toFixed(2),
        avg_profit_per_trade: Number(row.avg_profit_per_trade || 0).toFixed(2),
        avg_drawdown: Number(row.avg_drawdown || 0).toFixed(2)
      };
    }).sort((a, b) => Number(b.total_pnl) - Number(a.total_pnl));
  } catch (error) {
    console.error('Error fetching bot metrics:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);
    // Return an empty array instead of throwing to prevent the UI from breaking
    return [];
  }
}

export async function getBotById(botId) {
  try {
    // Build the query to get only needed columns
    const result = await query('SELECT * FROM sim_bots WHERE bot_id = $1', [botId]);
    
    if (result.rows.length === 0) {
      return null;
    }
    
    // Get only the most recent trades (limit to 20) with specific columns
    const tradesResult = await query(
      `SELECT 
         t.trade_id, t.bot_id, t.ticker, t.entry_price, t.exit_price, 
         t.trade_size, t.trade_direction, t.entry_time, t.exit_time, 
         t.trade_status, t.trade_pnl, t.pnl_percent, t.exit_trigger_price,
         b.name AS bot_name, b.algorithm_type
       FROM sim_bot_trades t
       JOIN sim_bots b ON t.bot_id = b.bot_id 
       WHERE t.bot_id = $1 
       ORDER BY t.entry_time DESC
       LIMIT 20`, 
      [botId]
    );
    
    // Efficiently map trade data
    const trades = tradesResult.rows.map(row => ({
      trade_id: row.trade_id,
      bot_id: row.bot_id,
      ticker: row.ticker,
      entry_price: Number(row.entry_price || 0),
      exit_price: Number(row.exit_price || 0),
      trade_size: Number(row.trade_size || 0),
      trade_direction: row.trade_direction,
      entry_time: row.entry_time,
      exit_time: row.exit_time,
      trade_status: row.trade_status,
      pnl: Number(row.trade_pnl || 0),
      pnl_percent: Number(row.pnl_percent || 0),
      trailing_stop_price: row.exit_trigger_price,
      exit_reason: row.exit_trigger_price ? 'trigger_price' : 'unknown',
      bot_name: row.bot_name,
      algorithm_type: row.algorithm_type
    }));
    
    // Get only required metrics columns
    const metricsResult = await query(
      `SELECT 
         bot_id, ticker, algo_id, avg_win_rate, profit_factor, 
         total_pnl, avg_profit_per_trade, avg_drawdown, max_drawdown,
         sharpe_ratio, r_multiple, total_trades
       FROM bot_metrics WHERE bot_id = $1`, 
      [botId]
    );
    
    let metrics = null;
    
    if (metricsResult.rows.length > 0) {
      const row = metricsResult.rows[0];
      const winRate = Number(row.avg_win_rate || 0) / 100; // Convert from percentage to decimal
      
      metrics = {
        bot_id: row.bot_id,
        ticker: row.ticker,
        win_rate: winRate.toFixed(4),
        profit_factor: Number(row.profit_factor || 0).toFixed(2),
        total_pnl: Number(row.total_pnl || 0).toFixed(2),
        average_win_amount: Number(row.avg_profit_per_trade || 0).toFixed(2),
        average_loss_amount: (-Math.abs(Number(row.avg_drawdown || 0))).toFixed(2),
        max_drawdown: (-Math.abs(Number(row.max_drawdown || 0))).toFixed(2),
        sharpe_ratio: Number(row.sharpe_ratio || 0).toFixed(2),
        risk_reward_ratio: Number(row.r_multiple || 0).toFixed(2),
        total_trades: Number(row.total_trades || 0),
        winning_trades: Math.round(Number(row.total_trades || 0) * winRate),
        losing_trades: Math.round(Number(row.total_trades || 0) * (1 - winRate)),
        rank_score: ((winRate * 0.5) + (Number(row.profit_factor || 1) * 0.3 / 3) + 
                   (Number(row.sharpe_ratio || 0) * 0.2 / 3)).toFixed(2)
      };
    }
    
    // Format the bot data with minimal property copying
    const bot = {
      bot_id: result.rows[0].bot_id,
      name: result.rows[0].name,
      ticker: result.rows[0].ticker,
      algorithm_module: result.rows[0].algorithm_module,
      algorithm_type: result.rows[0].algorithm_type,
      trade_direction: result.rows[0].trade_direction,
      position_size: Number(result.rows[0].position_size || 0),
      trailing_stop_pct: Number(result.rows[0].trailing_stop_pct || 0),
      description: result.rows[0].description || '',
      version: result.rows[0].version || '1.0',
      is_active: Boolean(result.rows[0].is_active),
      created_at: result.rows[0].created_at,
      last_updated: result.rows[0].last_updated,
      trades,
      metrics
    };
    
    return bot;
  } catch (error) {
    console.error(`Error fetching bot ${botId}:`, error);
    throw error;
  }
}

export async function getTickData(ticker = null, limit = 100) {
  try {
    console.log(`Fetching tick data${ticker ? ` for ${ticker}` : ''}`);
    
    let queryText = `
      SELECT id, ticker, timestamp, price, trade_size, volume, bid, ask
      FROM tick_data
    `;
    
    const queryParams = [];
    
    // Add ticker filter if provided
    if (ticker) {
      queryText += ' WHERE ticker = $1';
      queryParams.push(ticker);
    }
    
    // Add order and limit
    queryText += ' ORDER BY timestamp DESC LIMIT $' + (queryParams.length + 1);
    queryParams.push(limit);
    
    const result = await query(queryText, queryParams);
    console.log(`Retrieved ${result.rows.length} tick data records`);
    
    return result.rows.map(row => ({
      id: row.id,
      ticker: row.ticker,
      timestamp: row.timestamp,
      price: Number(row.price || 0),
      trade_size: Number(row.trade_size || 0),
      volume: Number(row.volume || 0),
      bid: Number(row.bid || 0),
      ask: Number(row.ask || 0)
    }));
  } catch (error) {
    console.error('Error fetching tick data:', error.message);
    return [];
  }
}

export default {
  getConnection,
  query,
  getBots,
  getTrades,
  getOpenTrades,
  getBotMetrics,
  getBotById,
  getTickData
};