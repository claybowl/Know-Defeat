import asyncpg
import asyncio
import pandas as pd

async def fetch_all_trades():
    # Connect to the database
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )

    async with db_pool.acquire() as conn:
        # Query all trades
        rows = await conn.fetch("""
            SELECT * FROM sim_bot_trades
        """)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Save to CSV
        df.to_csv('all_trades.csv', index=False)
        print("All trades have been saved to all_trades.csv")

    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(fetch_all_trades()) 