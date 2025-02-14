import asyncpg
import asyncio
import pandas as pd

async def fetch_tick_data():
    # Connect to the database
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )

    async with db_pool.acquire() as conn:
        # Query all tick data
        rows = await conn.fetch("""
            SELECT * FROM tick_data
        """)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Save to CSV
        df.to_csv('tick_data.csv', index=False)
        print("Tick data has been saved to tick_data.csv")

    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(fetch_tick_data()) 
