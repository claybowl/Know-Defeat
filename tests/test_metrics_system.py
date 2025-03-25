"""
Tests the metrics system to verify calculations are working properly
and that values are being saved to the database correctly.
"""

import asyncio
import asyncpg
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from src.metrics_calculator import MetricsCalculator
from src.metrics_updater import MetricsUpdater
from src.bot_ranker import BotRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def test_metrics_system():
    """Test the metrics calculation and storage system."""
    # Create a connection pool
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        # Create components
        metrics_calculator = MetricsCalculator(pool)
        metrics_updater = MetricsUpdater(pool, metrics_calculator)
        bot_ranker = BotRanker(pool)
        
        # Get all bots
        async with pool.acquire() as conn:
            bots = await conn.fetch("SELECT bot_id, ticker, algorithm_type FROM sim_bots")
            
            if not bots:
                logger.error("No bots found in the database.")
                return
            
            logger.info(f"Found {len(bots)} bots in the database")
            
            # Check if each bot has trades
            bot_trade_counts = []
            for bot in bots:
                bot_id = bot['bot_id']
                ticker = bot['ticker']
                
                # Count trades
                trade_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = $1",
                    bot_id
                )
                
                # Count open trades
                open_trades = await conn.fetchval(
                    "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = $1 AND trade_status = 'open'",
                    bot_id
                )
                
                # Get latest metrics
                metrics = await conn.fetchrow(
                    "SELECT * FROM bot_metrics WHERE bot_id = $1 ORDER BY timestamp DESC LIMIT 1",
                    bot_id
                )
                
                metrics_age = "No metrics" if not metrics else \
                    f"{(datetime.now() - metrics['timestamp']).total_seconds() / 60:.1f} min ago"
                
                bot_trade_counts.append({
                    'bot_id': bot_id,
                    'ticker': ticker,
                    'algorithm': bot['algorithm_type'],
                    'total_trades': trade_count,
                    'open_trades': open_trades,
                    'metrics_updated': metrics_age
                })
            
            # Display bot trade counts
            df = pd.DataFrame(bot_trade_counts)
            print("\n=== Bot Trade Summary ===")
            from tabulate import tabulate
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            
            # Update metrics for all bots that have trades
            print("\n=== Updating Metrics ===")
            for bot in bots:
                bot_id = bot['bot_id']
                ticker = bot['ticker']
                
                # Only update metrics for bots with trades
                trade_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sim_bot_trades WHERE bot_id = $1",
                    bot_id
                )
                
                if trade_count > 0:
                    logger.info(f"Updating metrics for bot {bot_id} ({ticker})")
                    result = await metrics_updater.update_bot_metrics(bot_id, ticker)
                    print(f"Bot {bot_id} ({ticker}): {'✅ Updated' if result else '❌ Failed'}")
                else:
                    print(f"Bot {bot_id} ({ticker}): ⚠️ Skipped (no trades)")
            
            # Rank the bots
            print("\n=== Ranking Bots ===")
            ranked_bots = await bot_ranker.rank_bots()
            
            if ranked_bots:
                # Display top 5 bots
                top_bots = ranked_bots[:min(5, len(ranked_bots))]
                
                rank_data = []
                for i, bot in enumerate(top_bots):
                    rank_data.append({
                        'Rank': i + 1,
                        'Bot ID': bot['bot_id'],
                        'Ticker': bot['ticker'],
                        'Score': f"{bot['rank_score']:.2f}",
                        'Win Rate': f"{bot.get('avg_win_rate', 0):.1f}%",
                        'PnL': f"${bot.get('total_pnl', 0):.2f}"
                    })
                
                print("\n=== Top 5 Ranked Bots ===")
                rank_df = pd.DataFrame(rank_data)
                from tabulate import tabulate
                print(tabulate(rank_df, headers='keys', tablefmt='psql', showindex=False))
            else:
                print("⚠️ No ranked bots found or ranking failed")
            
            # Check bot_metrics table status
            metrics_counts = await conn.fetch("""
                SELECT bot_id, COUNT(*) as record_count 
                FROM bot_metrics 
                GROUP BY bot_id 
                ORDER BY bot_id
            """)
            
            if metrics_counts:
                metrics_data = [{'Bot ID': m['bot_id'], 'Metric Records': m['record_count']} for m in metrics_counts]
                print("\n=== Bot Metrics Records ===")
                metrics_df = pd.DataFrame(metrics_data)
                from tabulate import tabulate
                print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))
            else:
                print("⚠️ No metrics records found in bot_metrics table")
    
    except Exception as e:
        logger.error(f"Error in metrics system test: {e}")
    
    finally:
        # Close the connection pool
        await pool.close()

if __name__ == "__main__":
    print("=== Bot Metrics System Test ===")
    print("This script checks your bot metrics system and updates metrics for bots with trades.")
    print("To run this script, make sure you have the required packages:")
    print("  - asyncpg: pip install asyncpg")
    print("  - pandas: pip install pandas")
    print("  - tabulate: pip install tabulate")
    print("\nFirst, ensure your PostgreSQL database is running.")
    print("Then run: conda activate Autogen && python tests/test_metrics_system.py")
    
    # Check for required packages
    missing_packages = []
    try:
        import asyncpg
    except ImportError:
        missing_packages.append("asyncpg")
    
    try:
        import pandas as pd
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        import tabulate
    except ImportError:
        missing_packages.append("tabulate")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install the required packages using:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        sys.exit(1)
    
    # Run the test if all packages are available
    try:
        asyncio.run(test_metrics_system())
    except Exception as e:
        print(f"\n❌ Error running metrics system test: {e}")
        print("Make sure your PostgreSQL database is running and accessible.")