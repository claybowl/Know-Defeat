import asyncio
import logging
from datetime import datetime, timedelta
import sys

sys.path.append('.')
from src.metrics_calculator import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_datetime_parameter_fixes():
    """Test that the datetime parameter fixes are working correctly."""
    logging.info("Starting datetime parameter fix verification...")
    
    # Initialize the metrics calculator
    calculator = MetricsCalculator()
    await calculator.init_db_pool()
    
    try:
        # List of bot IDs to test with
        bot_ids = [1, 2, 3, 4, 5]  # Add your actual bot IDs here
        algo_id = 1  # Replace with a valid algo_id
        
        # Test calculate_win_rate_over_period with timedelta
        logging.info("Testing calculate_win_rate_over_period with timedelta...")
        for bot_id in bot_ids:
            try:
                # Test with 1-day period
                win_rate_1d = await calculator.calculate_win_rate_over_period(
                    bot_id, algo_id, period=timedelta(days=1)
                )
                logging.info(f"Bot {bot_id}: 1-day win rate = {win_rate_1d}%")
                
                # Test with 7-day period
                win_rate_7d = await calculator.calculate_win_rate_over_period(
                    bot_id, algo_id, period=timedelta(days=7)
                )
                logging.info(f"Bot {bot_id}: 7-day win rate = {win_rate_7d}%")
                
                # Calculate price model score (which uses win_rate_over_period)
                price_score = await calculator.calculate_price_model_score(bot_id, algo_id)
                logging.info(f"Bot {bot_id}: Price model score = {price_score}")
                
                # Calculate price wall score (which also uses win_rate_over_period)
                wall_score = await calculator.calculate_price_wall_score(bot_id, algo_id)
                logging.info(f"Bot {bot_id}: Price wall score = {wall_score}")
                
            except Exception as e:
                logging.error(f"Error testing bot {bot_id}: {e}")
        
        # Test other functions that use timedelta
        logging.info("Testing other functions that use timedelta...")
        for bot_id in bot_ids:
            try:
                # Test one_day_performance
                one_day_perf = await calculator.calculate_one_day_performance(bot_id, algo_id)
                logging.info(f"Bot {bot_id}: 1-day performance = {one_day_perf}")
                
                # Test one_week_performance
                one_week_perf = await calculator.calculate_one_week_performance(bot_id, algo_id)
                logging.info(f"Bot {bot_id}: 1-week performance = {one_week_perf}")
                
                # Test trade_frequency
                daily_trades = await calculator.calculate_trade_frequency(bot_id, algo_id, timedelta(days=1))
                logging.info(f"Bot {bot_id}: Daily trade count = {daily_trades}")
                
            except Exception as e:
                logging.error(f"Error testing other timedelta functions for bot {bot_id}: {e}")
        
        logging.info("All datetime parameter tests completed!")
        
    except Exception as e:
        logging.error(f"Error during datetime parameter testing: {e}")
    finally:
        # Close the database pool
        await calculator.close_db_pool()

if __name__ == "__main__":
    asyncio.run(test_datetime_parameter_fixes()) 