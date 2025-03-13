import asyncio
import asyncpg

async def count_rows(pool, table_name):
    async with pool.acquire() as conn:
        # Count total rows
        count = await conn.fetchval(f'''
            SELECT COUNT(*)
            FROM {table_name}
        ''')
        print(f"The {table_name} table has {count} rows.")
        
        # Fetch most recent 20 rows
        rows = await conn.fetch(f'''
            SELECT *
            FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        
        print("\nMost recent 20 entries:")
        for row in rows:
            print(row)

async def main():
    pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',  
        database='tick_data',
        host='localhost'
    )

    await count_rows(pool, 'sim_bot_trades')

    await pool.close()

asyncio.run(main())
