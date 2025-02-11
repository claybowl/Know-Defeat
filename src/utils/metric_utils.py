from .db_utils import (
    execute_db_query,
    fetch_rows,
    create_db_pool,
    execute_query
)
import logging


def calculate_total_pnl(bot_id, algo_id):
    query = """
        SELECT SUM(trade_pnl) AS total_pnl
        FROM sim_bot_trades
        WHERE bot_id = %s AND algo_id = %s;
    """
    total_pnl = execute_db_query(query, (bot_id, algo_id))
    return total_pnl

def calculate_win_rate(bot_id, algo_id):
    query = """
        SELECT 
            COUNT(CASE WHEN profit_loss > 0 THEN 1 END)::FLOAT / 
            GREATEST(COUNT(*), 1) * 100 AS win_rate
        FROM sim_bot_trades
        WHERE bot_id = %s AND algo_id = %s;
    """
    return execute_db_query(query, (bot_id, algo_id))

# Similar functions for the other metrics...

async def update_bot_metrics(db_pool, bot_id):
    """Update performance metrics for a bot"""
    try:
        async with db_pool.acquire() as conn:
            # Calculate total PNL
            total_pnl = await conn.fetchval("""
                SELECT SUM(trade_pnl) 
                FROM sim_bot_trades 
                WHERE bot_id = $1 
                AND trade_status = 'closed'
            """, bot_id) or 0.0
            
            # Update metrics table
            await conn.execute("""
                INSERT INTO bot_metrics (bot_id, total_pnl, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (bot_id) DO UPDATE SET
                    total_pnl = EXCLUDED.total_pnl,
                    updated_at = EXCLUDED.updated_at
            """, bot_id, total_pnl)
            
    except Exception as e:
        logging.error(f"Error updating metrics: {e}")
        raise
