import asyncio
import asyncpg

async def count_rows(pool, table_name):
    async with pool.acquire() as conn:
        count = await conn.fetchval(f'''
            SELECT COUNT(*)
            FROM {table_name}
        ''')
        print(f"The {table_name} table has {count} rows.")

async def main():
    pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',  
        database='tick_data',
        host='localhost'
    )

    await count_rows(pool, 'tick_data')

    await pool.close()

asyncio.run(main())
