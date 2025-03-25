"""
Simple script to check if there are trades and metrics in the database.
"""

import psycopg2
import psycopg2.extras
import sys
import os
from datetime import datetime

# Configure the database connection
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

def check_metrics():
    """Check metrics and trades for all bots."""
    print("=== Checking Bot Metrics System ===")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get all bots
        cursor.execute("SELECT bot_id, ticker, algorithm_type FROM sim_bots ORDER BY bot_id")
        bots = cursor.fetchall()
        
        if not bots:
            print("No bots found in the database.")
            return
        
        print(f"Found {len(bots)} bots in the database")
        
        # Check if each bot has trades
        print("\n=== Bot Trade Summary ===")
        print("Bot ID | Ticker | Algorithm | Total Trades | Open Trades | Has Metrics")
        print("-" * 75)
        
        for bot in bots:
            bot_id = bot['bot_id']
            ticker = bot['ticker']
            algorithm = bot['algorithm_type']
            
            # Count trades
            cursor.execute(
                "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = %s",
                (bot_id,)
            )
            trade_count = cursor.fetchone()[0]
            
            # Count open trades
            cursor.execute(
                "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = %s AND trade_status = 'open'",
                (bot_id,)
            )
            open_trades = cursor.fetchone()[0]
            
            # Check if metrics exist
            cursor.execute(
                "SELECT COUNT(*) FROM bot_metrics WHERE bot_id = %s",
                (bot_id,)
            )
            metrics_count = cursor.fetchone()[0]
            
            has_metrics = "Yes" if metrics_count > 0 else "No"
            
            print(f"{bot_id:6} | {ticker:6} | {algorithm:15} | {trade_count:12} | {open_trades:10} | {has_metrics}")
        
        # Check total metrics count
        cursor.execute("SELECT COUNT(*) FROM bot_metrics")
        total_metrics = cursor.fetchone()[0]
        
        print(f"\nTotal metrics records in database: {total_metrics}")
        
        # Check latest trades
        cursor.execute("""
            SELECT trade_id, bot_id, ticker, trade_direction, entry_price, exit_price, 
                   trade_pnl, entry_time, exit_time
            FROM sim_bot_trades
            ORDER BY entry_time DESC
            LIMIT 5
        """)
        latest_trades = cursor.fetchall()
        
        if latest_trades:
            print("\n=== Latest Trades ===")
            print("Trade ID | Bot ID | Ticker | Direction | Entry Price | Exit Price | PnL | Entry Time | Exit Time")
            print("-" * 100)
            
            for trade in latest_trades:
                entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if trade['entry_time'] else 'N/A'
                exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if trade['exit_time'] else 'N/A'
                
                print(f"{trade['trade_id']:8} | {trade['bot_id']:6} | {trade['ticker']:6} | "
                      f"{trade['trade_direction']:9} | {trade['entry_price']:11.2f} | "
                      f"{trade['exit_price'] or 0:10.2f} | {trade['trade_pnl'] or 0:3.2f} | "
                      f"{entry_time} | {exit_time}")
        else:
            print("\nNo trades found in the database.")
    
    except psycopg2.Error as e:
        print(f"\nError connecting to database: {e}")
    
    except Exception as e:
        print(f"\nError checking metrics: {e}")
    
    finally:
        # Close the database connection
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_metrics()