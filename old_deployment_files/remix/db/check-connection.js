// Simple script to test database connectivity
const { Pool } = require('pg');

// Environment variables for connection
const isCloudSocket = process.env.CLOUD_SQL_CONNECTION_NAME && 
                      process.env.DB_HOST === '/cloudsql';

// Create connection config based on environment
let config;
if (isCloudSocket) {
  // Connect directly using UNIX socket
  const connectionName = process.env.CLOUD_SQL_CONNECTION_NAME;
  config = {
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'tick_data',
    host: `/cloudsql/${connectionName}`,
    max: 5,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000,
  };
  console.log(`Using Cloud SQL direct socket: /cloudsql/${connectionName}`);
} else {
  // Connect using TCP
  config = {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    database: process.env.DB_NAME || 'tick_data',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    max: 5,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000,
  };
  console.log(`Using TCP connection: ${process.env.DB_HOST || 'localhost'}:${process.env.DB_PORT || '5432'}`);
}

// Create a pool
const pool = new Pool(config);

async function testConnection() {
  console.log('Testing database connection...');
  
  try {
    // Acquire client from pool
    const client = await pool.connect();
    
    try {
      // Run simple query
      console.log('Querying database...');
      const result = await client.query('SELECT NOW() as time');
      
      console.log('✅ Successfully connected to database!');
      console.log(`Server time: ${result.rows[0].time}`);
      
      // Check for tables
      console.log('Checking for tables...');
      const tables = await client.query(`
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
      `);
      
      console.log(`Found ${tables.rowCount} tables:`);
      tables.rows.forEach(row => {
        console.log(`  - ${row.table_name}`);
      });
      
      // Check for bot data if tables exist
      if (tables.rows.some(row => row.table_name === 'sim_bots')) {
        console.log('Checking for bots...');
        const bots = await client.query('SELECT COUNT(*) as count FROM sim_bots');
        console.log(`Found ${bots.rows[0].count} bots in the database`);
      }
      
    } finally {
      // Release client back to pool
      client.release();
    }
  } catch (err) {
    console.error('❌ Connection failed:', err.message);
    console.error('Stack trace:', err.stack);
  } finally {
    // End pool
    await pool.end();
  }
}

// Run the test
testConnection();