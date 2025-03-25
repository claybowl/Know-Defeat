# test_bot_ranker.py
import pytest
import sys
import os
from unittest.mock import AsyncMock, patch

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from src
from src.bot_ranker import BotRanker

@pytest.fixture
def db_pool():
    # Mock database pool
    return AsyncMock()

@pytest.fixture
def bot_ranker(db_pool):
    return BotRanker(db_pool)

@pytest.mark.asyncio
async def test_fetch_bot_metrics(bot_ranker):
    # Mock database response
    bot_ranker.db_pool.acquire.return_value.__aenter__.return_value.fetch.return_value = [
        {'bot_id': 1, 'one_hour_performance': 10.0, 'two_hour_performance': 8.0}
    ]
    
    metrics = await bot_ranker.fetch_bot_metrics()
    assert len(metrics) == 1
    assert metrics[0]['bot_id'] == 1

@pytest.mark.asyncio
async def test_calculate_bot_rank(bot_ranker):
    weights = {'one_hour_performance': 10.0, 'two_hour_performance': 5.0}
    bot_metrics = {'one_hour_performance': 10.0, 'two_hour_performance': 8.0}
    
    rank_score = await bot_ranker.calculate_bot_rank(bot_metrics, weights)
    assert rank_score == 1.8  # Example expected score

@pytest.mark.asyncio
async def test_rank_bots(bot_ranker):
    # Mock methods
    bot_ranker.fetch_bot_metrics = AsyncMock(return_value=[
        {'bot_id': 1, 'one_hour_performance': 10.0},
        {'bot_id': 2, 'one_hour_performance': 5.0}
    ])
    bot_ranker.calculate_bot_rank = AsyncMock(side_effect=[2.0, 1.0])
    
    ranked_bots = await bot_ranker.rank_bots()
    assert ranked_bots[0]['bot_id'] == 1
    assert ranked_bots[1]['bot_id'] == 2

@pytest.mark.asyncio
async def test_update_bot_rankings(bot_ranker):
    # Mock database update
    bot_ranker.db_pool.acquire.return_value.__aenter__.return_value.execute = AsyncMock()
    
    ranked_bots = [{'bot_id': 1, 'rank_score': 2.0, 'rank': 1}]
    await bot_ranker._update_bot_rankings(ranked_bots)
    
    bot_ranker.db_pool.acquire.return_value.__aenter__.return_value.execute.assert_called_once()

# Add more tests as needed
