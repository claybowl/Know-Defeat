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
        # First, let's get the actual column names from the table
        table_info = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'sim_bot_trades' 
            ORDER BY ordinal_position;
        """)

        columns = [col['column_name'] for col in table_info]

        # Now query all trades with the actual columns
        rows = await conn.fetch(f"""
            SELECT * FROM sim_bot_trades
            ORDER BY trade_id
        """)

        # Convert to DataFrame with the correct column names
        df = pd.DataFrame(rows, columns=columns)

        # Save to CSV with headers
        df.to_csv('all_trades.csv', index=False)
        print("All trades have been saved to all_trades.csv")
        print(f"Columns in the exported file: {columns}")

    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(fetch_all_trades())