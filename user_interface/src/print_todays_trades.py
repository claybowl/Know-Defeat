import asyncpg
import asyncio
import pandas as pd
from datetime import datetime

async def fetch_todays_trades():
    # Connect to the database
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )

    async with db_pool.acquire() as conn:
        # Get today's date
        today = datetime.now().date()

        # Query today's trades
        rows = await conn.fetch("""
            SELECT * FROM sim_bot_trades
            WHERE trade_timestamp::date = $1
        """, today)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Save to CSV
        df.to_csv('todays_trades.csv', index=False)
        print("Today's trades have been saved to todays_trades.csv")

    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(fetch_todays_trades()) 