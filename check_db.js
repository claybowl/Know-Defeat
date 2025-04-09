/**
 * Simple script to test PostgreSQL database connectivity
 * 
 * Run with: node check_db.js
 */

const { Pool } = require('pg');

// Load environment variables from .env file if dotenv is available
try {
  require('dotenv').config();
} catch (err) {
  console.log('dotenv not available, using default connection params');
}

// Database connection parameters
const config = {
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432', 10),
  database: process.env.DB_NAME || 'tick_data',
  user: process.env.DB_USER || 'clayb',
  password: process.env.DB_PASSWORD || 'musicman',
  max: 1, // Max connections in pool
  idleTimeoutMillis: 5000, // Connection timeout
  connectionTimeoutMillis: 5000, // Connection attempt timeout
};

console.log('Checking database connection with parameters:');
console.log('Host:', config.host);
console.log('Port:', config.port);
console.log('Database:', config.database);
console.log('User:', config.user);
console.log('Password:', '*'.repeat(config.password.length)); // Don't log actual password

// Create a new pool
const pool = new Pool(config);

async function testConnection() {
  let client;
  
  try {
    console.log('Attempting to connect to the database...');
    client = await pool.connect();
    console.log('✅ Successfully connected to the database!');
    
    console.log('Testing query execution...');
    const result = await client.query('SELECT current_timestamp, current_database()');
    console.log('✅ Database query executed successfully');
    console.log('Current time:', result.rows[0].current_timestamp);
    console.log('Database name:', result.rows[0].current_database);
    
    // Try to access the sim_bots table
    console.log('Testing access to sim_bots table...');
    try {
      const botsResult = await client.query(
        'SELECT COUNT(*) AS bot_count FROM sim_bots'
      );
      console.log('✅ Found sim_bots table with', botsResult.rows[0].bot_count, 'records');
    } catch (err) {
      console.log('⚠️ Could not access sim_bots table:', err.message);
      console.log('This may be normal if the table doesn\'t exist yet.');
    }
    
    // Try to access the sim_bot_trades table
    console.log('Testing access to sim_bot_trades table...');
    try {
      const tradesResult = await client.query(
        'SELECT COUNT(*) AS trade_count FROM sim_bot_trades'
      );
      console.log('✅ Found sim_bot_trades table with', tradesResult.rows[0].trade_count, 'records');
    } catch (err) {
      console.log('⚠️ Could not access sim_bot_trades table:', err.message);
      console.log('This may be normal if the table doesn\'t exist yet.');
    }
    
  } catch (err) {
    console.error('❌ Database connection error:', err.message);
    console.error('Please check your database credentials and ensure PostgreSQL is running.');
  } finally {
    if (client) {
      client.release();
      console.log('Database client released');
    }
    await pool.end();
    console.log('Connection pool closed');
  }
}

testConnection();