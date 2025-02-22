import sys, os
# Adjust sys.path so that the "src" folder (which contains metrics_calculator.py) is in the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import asyncpg
import asyncio
import pandas as pd
from metrics_calculator import MetricsCalculator  # Now this should work correctly

async def import_trades_and_update_metrics():
    # Create a connection pool to the database
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman',
        database='tick_data',
        host='localhost'
    )
    
    # Optionally clear out existing trades so we avoid duplicates.
    async with db_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE sim_bot_trades;")
    
    # Load the CSV file (assumes all_trades.csv is in the current working directory)
    # Parse the datetime columns so that we get datetime objects rather than strings.
    df = pd.read_csv('all_trades.csv', parse_dates=['entry_time', 'exit_time', 'entry_trigger_time', 'exit_trigger_time'])
    
    # Convert the records to a list of dictionaries
    trades = df.to_dict(orient='records')
    
    # Prepare a query – adjust the columns based on your sim_bot_trades definition.
    insert_query = """
    INSERT INTO sim_bot_trades 
        (trade_id, bot_id, algo_id, ticker, entry_price, exit_price, entry_time, exit_time,
         entry_trigger_price, exit_trigger_price, entry_trigger_time, exit_trigger_time,
         trade_size, trade_direction, trade_pnl, trade_status)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
    """
    
    # Insert each trade from the CSV into sim_bot_trades.
    async with db_pool.acquire() as conn:
        for trade in trades:
            # If algo_id is missing or NaN, convert to None.
            algo_id = trade.get('algo_id')
            if pd.isnull(algo_id):
                algo_id = None

            # Convert datetime fields: if they are NaT, convert them to None.
            entry_time = trade.get('entry_time')
            if pd.isnull(entry_time):
                entry_time = None

            exit_time = trade.get('exit_time')
            if pd.isnull(exit_time):
                exit_time = None

            entry_trigger_time = trade.get('entry_trigger_time')
            if pd.isnull(entry_trigger_time):
                entry_trigger_time = None

            exit_trigger_time = trade.get('exit_trigger_time')
            if pd.isnull(exit_trigger_time):
                exit_trigger_time = None

            await conn.execute(
                insert_query,
                trade.get('trade_id'),
                trade.get('bot_id'),
                algo_id,
                trade.get('ticker'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                entry_time,
                exit_time,
                trade.get('entry_trigger_price'),
                trade.get('exit_trigger_price'),
                entry_trigger_time,
                exit_trigger_time,
                trade.get('trade_size'),
                trade.get('trade_direction'),
                trade.get('trade_pnl'),
                trade.get('trade_status')
            )
    
    # Build base bot_metrics records from distinct bots in sim_bot_trades.
    distinct_query = """
    SELECT DISTINCT bot_id, ticker, algo_id
    FROM sim_bot_trades;
    """
    
    async with db_pool.acquire() as conn:
        distinct_bots = await conn.fetch(distinct_query)
        for rec in distinct_bots:
            bot_id = rec['bot_id']
            ticker = rec['ticker']
            algo_id = rec['algo_id']
            # Insert a new bot_metrics record if it doesn't already exist. 
            # (bot_id is the primary key for bot_metrics in our create table.)
            try:
                await conn.execute("""
                  INSERT INTO bot_metrics (bot_id, ticker, algo_id, last_updated)
                  VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                  ON CONFLICT (bot_id) DO NOTHING;
                """, bot_id, ticker, algo_id)
            except Exception as e:
                print(f"Error inserting bot {bot_id}: {e}")
    
    # Update metrics for each bot using the metrics calculator functions.
    for rec in distinct_bots:
        bot_id = rec['bot_id']
        algo_id = rec['algo_id']
        try:
            await MetricsCalculator.calculate_and_insert_execution_metrics(db_pool, bot_id, algo_id)
            await MetricsCalculator.calculate_and_insert_win_streaks(db_pool, bot_id, algo_id)
        except Exception as e:
            print(f"Error updating metrics for bot {bot_id}: {e}")
    
    # Optional: Show the contents of bot_metrics once updates are complete.
    async with db_pool.acquire() as conn:
         metrics = await conn.fetch("SELECT * FROM bot_metrics;")
         print("Bot Metrics:")
         for m in metrics:
              print(dict(m))
    
    await db_pool.close()

if __name__ == '__main__':
    asyncio.run(import_trades_and_update_metrics()) 