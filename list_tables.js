/**
 * Script to list all tables in the database
 * Run with: node list_tables.js
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
  max: 1,
  idleTimeoutMillis: 5000,
  connectionTimeoutMillis: 5000,
};

console.log('Connecting to database:', config.database);
const pool = new Pool(config);

async function listTables() {
  let client;
  
  try {
    client = await pool.connect();
    console.log('Connected to database');
    
    // Get list of all tables
    const tableQuery = `
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY table_name;
    `;
    
    const result = await client.query(tableQuery);
    
    if (result.rows.length === 0) {
      console.log('No tables found in the database');
      return;
    }
    
    console.log(`Found ${result.rows.length} tables in the database:`);
    console.log('-------------------------------------');
    
    // For each table, show row count and structure
    for (const table of result.rows) {
      const tableName = table.table_name;
      console.log(`Table: ${tableName}`);
      
      // Get row count
      const countResult = await client.query(`SELECT COUNT(*) FROM ${tableName}`);
      console.log(`Row count: ${countResult.rows[0].count}`);
      
      // Get table structure
      const columnQuery = `
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position;
      `;
      
      const columnResult = await client.query(columnQuery, [tableName]);
      console.log('Columns:');
      columnResult.rows.forEach(column => {
        let columnType = column.data_type;
        if (column.character_maximum_length) {
          columnType += `(${column.character_maximum_length})`;
        }
        console.log(`  - ${column.column_name}: ${columnType}`);
      });
      
      // If table has fewer than 5 rows, show sample data
      if (parseInt(countResult.rows[0].count) > 0 && parseInt(countResult.rows[0].count) <= 5) {
        console.log('Sample data:');
        const sampleResult = await client.query(`SELECT * FROM ${tableName} LIMIT 1`);
        console.log(sampleResult.rows[0]);
      }
      
      console.log('-------------------------------------');
    }
    
  } catch (err) {
    console.error('Error:', err.message);
  } finally {
    if (client) {
      client.release();
    }
    await pool.end();
  }
}

listTables();