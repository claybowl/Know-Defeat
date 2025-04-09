// Simple script to verify database connection
const { Pool } = require('pg');

// Load environment variables
if (process.env.NODE_ENV !== 'production') {
  console.log('Loading environment variables from .env file');
  require('dotenv').config();
}

// Extract connection mode from command line arguments
const useCloudSql = process.argv.includes('cloud');
const useProxy = process.argv.includes('proxy');

// Get connection parameters from environment
const connectMode = useCloudSql ? 'Cloud SQL Socket' : (useProxy ? 'Cloud SQL Proxy' : 'Local PostgreSQL');
console.log(`Testing connection to: ${connectMode}`);

let connectionConfig;

if (useCloudSql && process.env.CLOUD_SQL_CONNECTION_NAME) {
  // Connect using Cloud SQL UNIX socket
  connectionConfig = {
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'tick_data',
    host: `/cloudsql/${process.env.CLOUD_SQL_CONNECTION_NAME}`,
  };
  console.log(`Using Cloud SQL socket: /cloudsql/${process.env.CLOUD_SQL_CONNECTION_NAME}`);
} else {
  // Connect using TCP
  connectionConfig = {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    database: process.env.DB_NAME || 'tick_data',
    user: process.env.DB_USER || 'clayb',
    password: process.env.DB_PASSWORD || 'musicman'
  };
  console.log(`Using TCP connection: ${connectionConfig.host}:${connectionConfig.port}`);
}

async function testConnection() {
  const pool = new Pool(connectionConfig);
  try {
    console.log('Connecting to database...');
    const client = await pool.connect();
    try {
      console.log('Connected successfully!');
      
      // Get PostgreSQL version
      const versionRes = await client.query('SELECT version()');
      console.log(`PostgreSQL version: ${versionRes.rows[0].version}`);
      
      // Get list of tables
      const tablesRes = await client.query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
      );
      
      console.log(`Found ${tablesRes.rows.length} tables:`);
      
      // For each table, get the count of rows
      for (const table of tablesRes.rows) {
        const countRes = await client.query(`SELECT COUNT(*) FROM ${table.table_name}`);
        console.log(`- ${table.table_name}: ${countRes.rows[0].count} rows`);
      }
      
      console.log('✅ Database connection test successful!');
    } finally {
      client.release();
    }
  } catch (err) {
    console.error('❌ Database connection failed!');
    console.error(err);
  } finally {
    await pool.end();
  }
}

testConnection().catch(console.error);