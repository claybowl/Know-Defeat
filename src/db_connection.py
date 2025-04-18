import asyncpg
import logging

logger = logging.getLogger(__name__)

# Global variable to hold the pool
pool = None

async def create_db_pool():
    try:
        return await asyncpg.create_pool(
            user='clayb',
            password='musicman',  # Replace with your actual password
            database='tick_data',
            host='localhost'
        )
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise

async def initialize_db_pool():
    """Initialize the global database pool."""
    global pool
    if pool is None:
        logger.info("Initializing database pool...")
        pool = await create_db_pool()
        logger.info("Database pool initialized.")
    return pool

async def close_db_pool():
    """Close the global database pool."""
    global pool
    if pool:
        logger.info("Closing database pool...")
        await pool.close()
        pool = None
        logger.info("Database pool closed.")

async def get_db_connection():
    """FastAPI dependency to get a database connection from the pool."""
    global pool
    if pool is None:
        raise RuntimeError("Database pool not initialized. Call initialize_db_pool() first.")
        
    conn = None
    try:
        conn = await pool.acquire()
        yield conn
    finally:
        if conn:
            await pool.release(conn)
