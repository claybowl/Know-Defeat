import asyncio
import logging

class TradeListener:
    def __init__(self, db_pool, metrics_updater):
        self.db_pool = db_pool
        self.metrics_updater = metrics_updater

    async def listen_for_trade_completion(self):
        while True:
            try:
                async with self.db_pool.acquire() as connection:
                    # Check for completed trades
                    completed_trades = await connection.fetch("""
                        SELECT trade_id, bot_id, ticker, trade_result
                        FROM trades
                        WHERE status = 'completed'
                        AND processed = FALSE;
                    """)
                    
                    for trade in completed_trades:
                        await self.process_trade(trade)
                    
                    # Sleep for a short period before checking again
                    await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Error in trade listener: {str(e)}")

    async def process_trade(self, trade):
        # Mark trade as processed
        async with self.db_pool.acquire() as connection:
            await connection.execute("""
                UPDATE trades
                SET processed = TRUE
                WHERE trade_id = $1;
            """, trade['trade_id'])
        
        # Trigger metric recalculation
        await self.recalculate_metrics(trade['bot_id'], trade['ticker'])
        
        # NEW: Trigger bot ranking update
        await self.update_bot_rankings(trade['bot_id'])

    # Add this new method to TradeListener class:

    async def update_bot_rankings(self, bot_id):
        try:
            # Create a BotRanker instance and rank the bots
            from bot_ranker import BotRanker
            ranker = BotRanker(self.db_pool)
            await ranker.rank_bots()
            logging.info(f"Updated bot rankings after processing trade for bot {bot_id}")
        except Exception as e:
            logging.error(f"Error updating bot rankings: {str(e)}")

    async def recalculate_metrics(self, bot_id, ticker):
        # Placeholder for metric recalculation logic
        logging.info(f"Recalculating metrics for bot {bot_id}, ticker {ticker}")
        # Possibly you want:
        await self.metrics_updater.update_bot_metrics(bot_id, ticker)
