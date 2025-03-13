import asyncio
import asyncpg

async def add_trade_pnl_column():
    # Connect to the database
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )

    try:
        async with db_pool.acquire() as conn:
            # Check if trade_pnl column exists
            column_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'sim_bot_trades' 
                    AND column_name = 'trade_pnl'
                );
            """)

            if not column_exists:
                # Add the trade_pnl column
                await conn.execute("""
                    ALTER TABLE sim_bot_trades 
                    ADD COLUMN trade_pnl DECIMAL;
                """)
                print("Successfully added trade_pnl column")
            else:
                print("trade_pnl column already exists")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(add_trade_pnl_column()) 