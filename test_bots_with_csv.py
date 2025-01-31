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
        # Construct a Pandas DataFrame with columns your bot expects
        ticks_df = pd.DataFrame([{
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        }])

        # Now pass this DataFrame into the bot's method
        if self.bot.position is not None:
            if self.bot.check_trailing_stop(price):
                await self.bot.execute_trade("SELL", price, timestamp)
        else:
            # Make sure the bot's expected columns match the columns below
            if self.bot.analyze_price_conditions(ticks_df):
                await self.bot.execute_trade("BUY", price, timestamp)

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

            # Print results every 15 seconds instead of every minute
            if datetime.now() >= self.start_time + timedelta(seconds=15):
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

async def test_all_bots(csv_file):
    # Create each bot and a corresponding BotTester instance
    bots = [
        CoinLongBot(None, None, 'coin_long_bot'),
        CoinShortBot(None, None, 'coin_short_bot'),
        TSLALongBot(None, None, 'tsla_long_bot'),
        TSLAShortBot(None, None, 'tsla_short_bot')
    ]

    # Run each bot through the BotTester against the provided csv file
    for bot in bots:
        current_tester = BotTester(bot)
        print(f"Backtesting {bot.__class__.__name__} with {csv_file}")
        await current_tester.run(csv_file)  # Runs the trade logic over CSV data

if __name__ == "__main__":
    import sys

    # If a CSV filename is passed as an argument, use it. Otherwise default to 'todays_tick_data.csv'
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = 'todays_tick_data.csv'

    # Run backtests for all bots
    asyncio.run(test_all_bots(input_csv))

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
