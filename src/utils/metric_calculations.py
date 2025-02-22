import asyncpg
from datetime import datetime, timedelta

async def calculate_performance_over_period(db_pool, bot_id, algo_id, period):
    async with db_pool.acquire() as connection:
        end_time = datetime.now()
        start_time = end_time - period
        result = await connection.fetchval("""
            SELECT SUM(trade_pnl) 
            FROM sim_bot_trades
            WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed' 
                  AND timestamp BETWEEN $3 AND $4;
        """, bot_id, algo_id, start_time, end_time)
        
        # Return formatted value if result is not None, else return 0
        return round(result, 2) if result else 0

async def calculate_one_day_performance(db_pool, bot_id, algo_id):
    return await calculate_performance_over_period(db_pool, bot_id, algo_id, timedelta(days=1))

async def calculate_one_week_performance(db_pool, bot_id, algo_id):
    return await calculate_performance_over_period(db_pool, bot_id, algo_id, timedelta(weeks=1))

async def calculate_one_month_performance(db_pool, bot_id, algo_id):
    # Assuming a month of 30 days
    return await calculate_performance_over_period(db_pool, bot_id, algo_id, timedelta(days=30))

async def calculate_profit_per_second(db_pool, bot_id, algo_id):
    async with db_pool.acquire() as connection:
        # Calculate total PnL and total time span
        total_pnl = await connection.fetchval("""
            SELECT SUM(trade_pnl)
            FROM sim_bot_trades
            WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed';
        """, bot_id, algo_id)
        
        time_span_result = await connection.fetchrow("""
            SELECT MIN(timestamp) as start_time, MAX(timestamp) as end_time
            FROM sim_bot_trades
            WHERE bot_id = $1 AND algo_id = $2 AND trade_status = 'closed';
        """, bot_id, algo_id)
        
        if time_span_result and total_pnl is not None:
            start_time, end_time = time_span_result['start_time'], time_span_result['end_time']
            total_seconds = (end_time - start_time).total_seconds()
            profit_per_second = total_pnl / total_seconds if total_seconds > 0 else 0
            return round(profit_per_second, 2)
        else:
            return 0
