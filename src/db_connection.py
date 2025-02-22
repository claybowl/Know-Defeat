import asyncpg

async def create_db_pool():
    return await asyncpg.create_pool(
        user='clayb',
        password='musicman',  # Replace with your actual password
        database='tick_data',
        host='localhost'
    )
