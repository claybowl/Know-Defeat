#!/usr/bin/env python3
"""
GCP Database Connection Test Script
This script tests the connection to the GCP Cloud SQL database.
"""

import asyncio
import asyncpg
import logging
import os
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

async def test_database_connection():
    """Test connection to the GCP Cloud SQL database."""
    logger.info("Testing GCP Cloud SQL database connection...")
    
    try:
        # Get connection details from environment variables
        db_user = os.environ.get('DB_USER', 'postgres')
        db_password = os.environ.get('DB_PASSWORD', '')
        db_name = os.environ.get('DB_NAME', 'tick_data')
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = int(os.environ.get('DB_PORT', '5432'))
        
        logger.info(f"Connecting to {db_host}:{db_port} as {db_user}")
        
        # Create connection pool
        pool = await asyncpg.create_pool(
            user=db_user,
            password=db_password,
            database=db_name,
            host=db_host,
            port=db_port
        )
        
        # Test query
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"Connected to PostgreSQL: {version}")
            
            # Test tables
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            logger.info(f"Found {len(tables)} tables:")
            for table in tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
                logger.info(f"  - {table['table_name']}: {count} rows")
        
        await pool.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    db_success = await test_database_connection()
    if db_success:
        logger.info("✅ Successfully connected to GCP Cloud SQL database!")
    else:
        logger.error("❌ Failed to connect to GCP Cloud SQL database")

if __name__ == "__main__":
    asyncio.run(main())