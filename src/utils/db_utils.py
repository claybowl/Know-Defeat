# db_utils.py
import asyncpg
import logging

async def create_db_pool(user, password, database, host, port, min_size, max_size):
    """Create a connection pool to the database."""
    try:
        pool = await asyncpg.create_pool(
            user=user,
            password=password, 
            database=database,
            host=host,
            port=port,
            min_size=min_size,
            max_size=max_size
        )
        print("Database connection pool created successfully")
        return pool
    except Exception as e:
        print(f"Error creating database connection pool: {e}")
        raise

async def execute_query(db_pool, query, *args):
    """Execute a SQL query with optional arguments."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(query, *args)
    except Exception as e:
        print(f"Error executing query: {e}")
        raise

async def fetch_rows(db_pool, query, *args):
    """Fetch rows from the database using a SQL query with optional arguments."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return rows
    except Exception as e:
        print(f"Error fetching rows: {e}")
        raise

async def create_db_pool():
    return await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )

async def execute_db_query(query, params=None):
    pool = await create_db_pool()
    async with pool.acquire() as conn:
        result = await conn.fetch(query, *(params or []))
    await pool.close()
    return result
