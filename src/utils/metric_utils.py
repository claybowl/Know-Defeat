from src.utils.db_utils import execute_db_query
from src.utils.db_utils import fetch_rows
from src.utils.db_utils import create_db_pool
from src.utils.db_utils import execute_query
from src.utils.db_utils import db_pool
from ib_controller import IBDataIngestion


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

def update_bot_metrics(bot_id, algo_id):
    total_pnl = calculate_total_pnl(bot_id, algo_id)
    win_rate = calculate_win_rate(bot_id, algo_id)
    # Calculate other metrics...
    
    # Insert or update the bot_metrics table
    upsert_query = """
        INSERT INTO bot_metrics (bot_id, algo_id, total_pnl, win_rate, ...)
        VALUES (%s, %s, %s, %s, ...)
        ON CONFLICT (bot_id, algo_id) DO UPDATE
        SET total_pnl = %s, win_rate = %s, ...
    """
    execute_db_query(upsert_query, (bot_id, algo_id, total_pnl, win_rate, ..., total_pnl, win_rate, ...))