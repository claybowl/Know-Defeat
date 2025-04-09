// cloud-db.server.js
import pkg from 'pg';
const { Pool } = pkg;
import { mockData } from './db.server';

// Flag to use mock data instead of real database
// Controlled by environment variable
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
      user: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || '',
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 10000,
    });
    console.log(`Connecting to Cloud SQL using TCP: ${process.env.DB_HOST || 'localhost'}:${process.env.DB_PORT || '5432'}`);
  }
}

export async function getConnection() {
  if (USE_MOCK_DATA) {
    console.log("Using mock data (connection)");
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
    // Fallback to mock data
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
    return mockData.metrics;
  }
  
  try {
    // First try with rank_score (if column exists)
    const result = await query('SELECT * FROM bot_metrics ORDER BY rank_score DESC');
    return result.rows;
  } catch (error) {
    if (error.message.includes('column "rank_score" does not exist')) {
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
    throw error; // rethrow if it's some other error
  }
}

export async function getBotById(botId) {
  if (USE_MOCK_DATA) {
    const bot = mockData.bots.find(b => b.bot_id === parseInt(botId));
    if (!bot) return null;
    
    // Get bot trades
    const trades = mockData.trades.filter(t => t.bot_id === parseInt(botId));
    
    // Get bot metrics
    const metrics = mockData.metrics.find(m => m.bot_id === parseInt(botId));
    
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

export default {
  getConnection,
  query,
  getBots,
  getTrades,
  getOpenTrades,
  getBotMetrics,
  getBotById,
};