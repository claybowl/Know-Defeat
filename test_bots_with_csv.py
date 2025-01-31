import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from src.bots.COIN_long_bot import CoinLongBot
from src.bots.COIN_short_bot import CoinShortBot
from src.bots.TSLA_long_bot import TSLALongBot
from src.bots.TSLA_short_bot import TSLAShortBot

class BotTester:
    def __init__(self, bot):
        self.bot = bot
        self.trades = []
        self.start_time = datetime.now()

    async def mock_process_tick(self, timestamp, price, volume):
        # Simulate processing a tick
        print(f"Processing tick at {timestamp} with price {price} and volume {volume}")
        # Simulate trade logic
        trade = {'timestamp': timestamp, 'price': price, 'pnl': 0}  # Example trade structure
        self.trades.append(trade)

    async def run(self, csv_file):
        # Read the CSV file
        df = pd.read_csv(csv_file, header=None, names=['id', 'ticker', 'timestamp', 'price', 'trade_size', 'bid', 'ask', 'created_at', 'volume'])
        print(df.columns)  # Print column names for debugging

        # Iterate over the DataFrame rows
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            price = row['price']
            volume = row['volume']
            await self.mock_process_tick(timestamp, price, volume)
            await asyncio.sleep(0.1)  # Simulate a delay between ticks

            # Print results every minute
            if datetime.now() >= self.start_time + timedelta(minutes=1):
                self.report_results()
                self.start_time = datetime.now()

        self.report_results()

    def report_results(self):
        # Example reporting logic
        total_trades = len(self.trades)
        print(f"Total trades executed: {total_trades}")
        if total_trades > 0:
            pnl = sum(trade['pnl'] for trade in self.trades)
            print(f"Total PnL: {pnl}")
            print(f"Average PnL per trade: {pnl / total_trades}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        filename='bot_test.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize your bot
    # Replace with the bot you want to test
    test_bot = TSLALongBot(None, None, '1_bot')  # Adjust parameters as needed

    # Create a tester instance
    tester = BotTester(test_bot)

    # Run the tester with the CSV file
    asyncio.run(tester.run('todays_tick_data.csv'))
