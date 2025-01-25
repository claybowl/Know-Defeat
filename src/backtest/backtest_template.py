import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
import pytz
from decimal import Decimal
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import the bot file you want to test
from bots.TSLA_short_bot import TSLAShortBot

class Backtester:
    def __init__(self, db_pool, start_date, end_date, bot_id):
        self.bot_id = bot_id
        self.db_pool = db_pool
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        self.trades = []  # Store the trades made by the bot

    async def get_historical_ticks(self):
        """Fetch historical tick data from the database"""
        async with self.db_pool.acquire() as conn:
            ticks = await conn.fetch('''
                SELECT ticker, timestamp, price, volume 
                FROM tick_data
                WHERE 
                    ticker = 'COIN'  -- Change this if your bot trades a different ticker
                    AND timestamp BETWEEN $1 AND $2
                ORDER BY timestamp ASC
            ''', self.start_date, self.end_date)
            return ticks

    async def run_backtest(self):
        """Execute the backtest using historical data"""
        # Initialize the bot
        bot = TSLAShortBot(self.db_pool)  # Change this line to match the bot you imported
        
        # Add a method to the bot to store trades
        async def log_trade(trade):
            self.trades.append(trade)
        bot.log_trade = log_trade
        
        # Get historical data
        ticks = await self.get_historical_ticks()
        
        # Process each tick through the bot
        for tick in ticks:
            await bot.process_tick(tick['ticker'], tick['price'], tick['timestamp'])

    def analyze_results(self):
        """Analyze the results of the backtest"""
        print(f"\nBacktest Results for {self.start_date} to {self.end_date}")
        print(f"Number of Trades: {len(self.trades)}")
        
        if len(self.trades) == 0:
            print("No trades were made during the backtest period.")
            return
        
        total_profit = sum(trade['profit_loss'] for trade in self.trades)
        print(f"Total Profit/Loss: ${total_profit:.2f}")
        
        winning_trades = [trade for trade in self.trades if trade['profit_loss'] > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100
        print(f"Win Rate: {win_rate:.2f}%")
        
        if len(winning_trades) > 0:
            avg_profit = sum(trade['profit_loss'] for trade in winning_trades) / len(winning_trades)
            print(f"Average Profit per Winning Trade: ${avg_profit:.2f}")
        
        losing_trades = [trade for trade in self.trades if trade['profit_loss'] < 0]
        if len(losing_trades) > 0:
            avg_loss = sum(trade['profit_loss'] for trade in losing_trades) / len(losing_trades)
            print(f"Average Loss per Losing Trade: ${avg_loss:.2f}")

async def main():
    # Setup database connection
    db_pool = await asyncpg.create_pool(
        user='clayb',
        password='musicman', 
        database='tick_data',
        host='localhost'
    )

    # Define backtest period
    start_date = datetime(2023, 1, 1)  # Update to your desired start date
    end_date = datetime(2023, 12, 31)  # Update to your desired end date
    
    # Create and run backtester
    bot_id = 1  # Replace with the actual bot_id you want to test
    backtester = Backtester(db_pool, start_date, end_date, bot_id)
    await backtester.run_backtest()
    
    # Analyze and print results
    backtester.analyze_results()

    # Clean up
    await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
