import logging
import asyncpg
from decimal import Decimal

class BotRanker:
    """
    Enhanced bot ranking system that uses weighted metrics from variable_weights table.
    Incorporates all available metrics in bot_metrics table and allows for dynamic
    adjustment of weights.
    """
    
    def __init__(self, db_pool):
        """Initialize with a database connection pool."""
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)

    async def get_variable_weights(self):
        """Fetch current variable weights from the database."""
        try:
            async with self.db_pool.acquire() as connection:
                weights = await connection.fetch("""
                    SELECT variable_name, weight 
                    FROM variable_weights 
                    ORDER BY variable_name
                """)
                
                # Convert to dictionary for easier access
                weights_dict = {row['variable_name']: row['weight'] for row in weights}
                return weights_dict
        except Exception as e:
            self.logger.error(f"Error fetching variable weights: {e}")
            return {}

    async def fetch_bot_metrics(self):
        """Fetch the latest metrics for all bots."""
        try:
            async with self.db_pool.acquire() as connection:
                # Get the latest metrics for each bot
                metrics = await connection.fetch("""
                    SELECT DISTINCT ON (bot_id) 
                        bot_id, 
                        ticker,
                        algorithm_id,
                        one_hour_performance,
                        two_hour_performance,
                        one_day_performance,
                        one_week_performance,
                        one_month_performance,
                        avg_win_rate,
                        avg_drawdown,
                        profit_per_second,
                        price_model_score,
                        volume_model_score,
                        price_wall_score,
                        win_streak_2,
                        win_streak_3,
                        win_streak_4,
                        win_streak_5,
                        win_streak_6,
                        win_streak_7,
                        timestamp
                    FROM bot_metrics
                    ORDER BY bot_id, timestamp DESC
                """)
                
                return metrics
        except Exception as e:
            self.logger.error(f"Error fetching bot metrics: {e}")
            return []

    async def calculate_bot_rank(self, bot_metrics, weights):
        """
        Calculate a weighted rank score for a single bot based on its metrics 
        and the current variable weights.
        """
        try:
            # Start with zero score
            score = Decimal('0.0')
            
            # Process each metric that has a corresponding weight
            if 'one_hour_performance' in weights and bot_metrics['one_hour_performance'] is not None:
                score += Decimal(str(weights['one_hour_performance'])) * Decimal(str(bot_metrics['one_hour_performance']))
                
            if 'two_hour_performance' in weights and bot_metrics['two_hour_performance'] is not None:
                score += Decimal(str(weights['two_hour_performance'])) * Decimal(str(bot_metrics['two_hour_performance']))
                
            if 'one_day_performance' in weights and bot_metrics['one_day_performance'] is not None:
                score += Decimal(str(weights['one_day_performance'])) * Decimal(str(bot_metrics['one_day_performance']))
                
            if 'one_week_performance' in weights and bot_metrics['one_week_performance'] is not None:
                score += Decimal(str(weights['one_week_performance'])) * Decimal(str(bot_metrics['one_week_performance']))
                
            if 'one_month_performance' in weights and bot_metrics['one_month_performance'] is not None:
                score += Decimal(str(weights['one_month_performance'])) * Decimal(str(bot_metrics['one_month_performance']))
                
            if 'avg_win_rate' in weights and bot_metrics['avg_win_rate'] is not None:
                score += Decimal(str(weights['avg_win_rate'])) * Decimal(str(bot_metrics['avg_win_rate']))
                
            if 'avg_drawdown' in weights and bot_metrics['avg_drawdown'] is not None:
                # Invert drawdown since lower is better
                score += Decimal(str(weights['avg_drawdown'])) * (Decimal('100.0') - Decimal(str(bot_metrics['avg_drawdown'])))
                
            if 'profit_per_second' in weights and bot_metrics['profit_per_second'] is not None:
                score += Decimal(str(weights['profit_per_second'])) * Decimal(str(bot_metrics['profit_per_second']))
                
            if 'price_model_score' in weights and bot_metrics['price_model_score'] is not None:
                score += Decimal(str(weights['price_model_score'])) * Decimal(str(bot_metrics['price_model_score']))
                
            if 'volume_model_score' in weights and bot_metrics['volume_model_score'] is not None:
                score += Decimal(str(weights['volume_model_score'])) * Decimal(str(bot_metrics['volume_model_score']))
                
            if 'price_wall_score' in weights and bot_metrics['price_wall_score'] is not None:
                score += Decimal(str(weights['price_wall_score'])) * Decimal(str(bot_metrics['price_wall_score']))
                
            # Process win streak metrics
            for i in range(2, 8):
                streak_key = f'win_streak_{i}'
                if streak_key in weights and bot_metrics[streak_key] is not None:
                    score += Decimal(str(weights[streak_key])) * Decimal(str(bot_metrics[streak_key]))
            
            # Normalize score (divide by 100)
            score = score / Decimal('100.0')
            
            return float(score)
        except Exception as e:
            self.logger.error(f"Error calculating bot rank: {e}")
            return 0.0

    async def rank_bots(self):
        """
        Rank all bots based on their metrics and the current variable weights.
        Returns a list of bots sorted by rank (highest first).
        """
        try:
            # Get current weights
            weights = await self.get_variable_weights()
            if not weights:
                self.logger.warning("No weights found, using default ranking method")
                return await self._default_rank_bots()
            
            # Get bot metrics
            metrics = await self.fetch_bot_metrics()
            if not metrics:
                self.logger.warning("No bot metrics found")
                return []
            
            # Calculate rank for each bot
            ranked_bots = []
            for bot in metrics:
                # Convert to regular dict for easier manipulation
                bot_dict = dict(bot)
                
                # Calculate rank score
                rank_score = await self.calculate_bot_rank(bot_dict, weights)
                
                # Add rank score to the bot data
                bot_dict['rank_score'] = rank_score
                ranked_bots.append(bot_dict)
            
            # Sort by rank score (descending)
            ranked_bots.sort(key=lambda x: x['rank_score'], reverse=True)
            
            # Update rankings in database
            await self._update_bot_rankings(ranked_bots)
            
            return ranked_bots
        except Exception as e:
            self.logger.error(f"Error ranking bots: {e}")
            return []

    async def _default_rank_bots(self):
        """
        Fallback method to rank bots without using the variable_weights table.
        Uses a simple sum of one_hour_performance and avg_win_rate.
        """
        async with self.db_pool.acquire() as connection:
            bots = await connection.fetch("""
                SELECT DISTINCT ON (bot_id) 
                    bot_id, ticker, one_hour_performance, avg_win_rate, timestamp
                FROM bot_metrics
                ORDER BY bot_id, timestamp DESC
            """)

            # Convert to dict and calculate simple rank
            ranked_bots = []
            for bot in bots:
                bot_dict = dict(bot)
                
                # Simple score calculation
                one_hour = bot_dict.get('one_hour_performance', 0) or 0
                win_rate = bot_dict.get('avg_win_rate', 0) or 0
                bot_dict['rank_score'] = one_hour + win_rate
                
                ranked_bots.append(bot_dict)

            # Sort by simple score
            ranked_bots.sort(key=lambda bot: bot['rank_score'], reverse=True)
            return ranked_bots

    async def _update_bot_rankings(self, ranked_bots):
        """Update the bot_rankings table with the latest rankings."""
        try:
            async with self.db_pool.acquire() as connection:
                # Check if table exists, create if not
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS bot_rankings (
                        ranking_id SERIAL PRIMARY KEY,
                        bot_id INTEGER NOT NULL,
                        rank_score DECIMAL(10,2) NOT NULL,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT true,
                        UNIQUE(bot_id)
                    )
                """)
                
                # Update rankings for each bot
                for index, bot in enumerate(ranked_bots):
                    await connection.execute("""
                        INSERT INTO bot_rankings (bot_id, rank_score, timestamp)
                        VALUES ($1, $2, NOW())
                        ON CONFLICT (bot_id) 
                        DO UPDATE SET 
                            rank_score = $2,
                            timestamp = NOW()
                    """, bot['bot_id'], bot['rank_score'])
        except Exception as e:
            self.logger.error(f"Error updating bot rankings: {e}")

    async def get_fund_allocation(self, total_funds, max_allocation_pct=10.0):
        """
        Calculate fund allocation based on bot rankings.
        
        Args:
            total_funds: Total funds available for trading
            max_allocation_pct: Maximum percentage to allocate to a single bot (default: 10%)
            
        Returns:
            List of dicts with bot_id, ticker, rank_score, and allocation_amount
        """
        try:
            # Get ranked bots
            ranked_bots = await self.rank_bots()
            if not ranked_bots:
                return []
            
            # Get active bots
            async with self.db_pool.acquire() as connection:
                active_bots = await connection.fetch("""
                    SELECT bot_id FROM bot_rankings
                    WHERE is_active = true
                    ORDER BY rank_score DESC
                """)
                
                active_bot_ids = [row['bot_id'] for row in active_bots]
            
            # Filter ranked bots to only include active ones
            active_ranked_bots = [bot for bot in ranked_bots if bot['bot_id'] in active_bot_ids]
            
            # Calculate allocations
            max_per_bot = total_funds * (max_allocation_pct / 100.0)
            total_score = sum(bot['rank_score'] for bot in active_ranked_bots)
            
            allocations = []
            remaining_funds = total_funds
            
            # First pass: allocate proportionally up to max_per_bot
            for bot in active_ranked_bots:
                if total_score > 0:
                    raw_allocation = (bot['rank_score'] / total_score) * total_funds
                    allocation = min(raw_allocation, max_per_bot)
                else:
                    # If total score is 0, allocate equally
                    allocation = min(total_funds / len(active_ranked_bots), max_per_bot)
                
                allocations.append({
                    'bot_id': bot['bot_id'],
                    'ticker': bot['ticker'],
                    'rank_score': bot['rank_score'],
                    'allocation_amount': allocation
                })
                
                remaining_funds -= allocation
            
            # Second pass: If there are remaining funds, redistribute them
            if remaining_funds > 0 and allocations:
                # Redistribute equally among bots that aren't at max allocation
                non_maxed_bots = [a for a in allocations if a['allocation_amount'] < max_per_bot]
                
                if non_maxed_bots:
                    additional_per_bot = remaining_funds / len(non_maxed_bots)
                    
                    for alloc in allocations:
                        if alloc['allocation_amount'] < max_per_bot:
                            new_amount = min(alloc['allocation_amount'] + additional_per_bot, max_per_bot)
                            additional = new_amount - alloc['allocation_amount']
                            alloc['allocation_amount'] = new_amount
                            remaining_funds -= additional
            
            return allocations
        except Exception as e:
            self.logger.error(f"Error calculating fund allocation: {e}")
            return []

    async def toggle_bot_active_status(self, bot_id, is_active):
        """Toggle a bot's active status for trading."""
        try:
            async with self.db_pool.acquire() as connection:
                await connection.execute("""
                    UPDATE bot_rankings
                    SET is_active = $2
                    WHERE bot_id = $1
                """, bot_id, is_active)
                
                return True
        except Exception as e:
            self.logger.error(f"Error toggling bot active status: {e}")
            return False
