import asyncio
import logging
from typing import Optional
from datetime import datetime

from src.db_connection import create_db_pool
from src.metrics_calculator_improvements import EnhancedMetricsCalculator
from src.enhanced_metrics_updater import EnhancedMetricsUpdater
from src.bot_ranker import BotRanker

logger = logging.getLogger("PerformancePoller")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PerformancePoller:
    """
    Real-time polling + metrics update and ranking system.
    """
    def __init__(self, poll_interval_seconds: int = 60):
        self.poll_interval_seconds = poll_interval_seconds
        self.db_pool = None
        self.metrics_calculator = None
        self.metrics_updater = None
        self.bot_ranker = None
        self.shutdown_event = asyncio.Event()
        self.running = False

    async def initialize(self):
        logger.info("Initializing PerformancePoller async components...")
        self.db_pool = await create_db_pool()
        self.metrics_calculator = EnhancedMetricsCalculator(self.db_pool)
        self.metrics_updater = EnhancedMetricsUpdater(self.db_pool, self.metrics_calculator)
        self.bot_ranker = BotRanker(self.db_pool)
        logger.info("PerformancePoller initialization complete.")

    async def poll_once(self):
        logger.info("Starting polling tick...")
        try:
            async with self.db_pool.acquire() as conn:
                # Get all active bots
                bot_rows = await conn.fetch("SELECT bot_id, ticker FROM sim_bots WHERE is_active = TRUE")

            # Iterate over all bots and update their metrics
            updated_bots = 0
            for row in bot_rows:
                bot_id = row["bot_id"]
                ticker = row["ticker"]
                success = await self.metrics_updater.update_bot_metrics(bot_id, ticker)
                if success:
                    logger.info(f"Updated metrics for bot {bot_id} ({ticker})")
                    updated_bots += 1
                else:
                    logger.error(f"Failed to update metrics for bot {bot_id} ({ticker})")

            logger.info(f"Metrics updated for {updated_bots}/{len(bot_rows)} active bots.")

            # Re-rank bots after update
            await self.bot_ranker.rerank_bots()

            # Optionally:
            # - check performance to enable/disable bots
            # - record poll outcome/history
            logger.info("Polling tick complete.")
        except Exception as e:
            logger.error(f"Error in poll_once: {e}")

    async def run(self):
        await self.initialize()
        logger.info("PerformancePoller started.")
        self.running = True
        try:
            while not self.shutdown_event.is_set():
                await self.poll_once()
                # Correctly create a task for the shutdown event wait
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                await asyncio.wait(
                    [shutdown_task],
                    timeout=self.poll_interval_seconds,
                )
                # Cancel the task if it didn't complete (i.e., timeout occurred)
                if not shutdown_task.done():
                    shutdown_task.cancel()
        finally:
            logger.info("PerformancePoller stopped.")
            self.running = False
            await self.teardown()

    async def stop(self):
        logger.info("Received shutdown signal.")
        self.shutdown_event.set()

    async def teardown(self):
        if self.db_pool is not None:
            await self.db_pool.close()
            logger.info("Database pool closed.")


# Entrypoint, so you can run as a script
if __name__ == "__main__":
    import signal

    async def main():
        poller = PerformancePoller(poll_interval_seconds=60)  # Adjust interval as desired

        def handle_signal(*args):
            asyncio.create_task(poller.stop())
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, handle_signal)
        loop.add_signal_handler(signal.SIGTERM, handle_signal)

        await poller.run()

        asyncio.run(main())
