import asyncio
import asyncpg
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

async def run_ranking():
    try:
        from src.bot_ranker import BotRanker
        
        # Database connection
        pool = await asyncpg.create_pool(
            user="clayb",
            password="musicman",
            database="tick_data",
            host="localhost"
        )
        
        # Create ranker and run ranking
        ranker = BotRanker(pool)
        ranked_bots = await ranker.rank_bots()
        
        print(f"Successfully ranked {len(ranked_bots)} bots")
        
        # Check if any data was inserted
        result = await pool.fetch("SELECT * FROM bot_rankings")
        print(f"Found {len(result)} records in bot_rankings table")
        
        await pool.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_ranking()) 