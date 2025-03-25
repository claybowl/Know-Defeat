import logging
import asyncpg
from decimal import Decimal
from src.trade_manager import TradeManager

class BotRanker:
    """
    Enhanced bot ranking system that uses weighted metrics from variable_weights table.
    Incorporates all available metrics in bot_metrics table and allows for dynamic
    adjustment of weights.
    
    Now integrated with TradeManager for dynamic trade-based allocation strategy.
    """
    
    def __init__(self, db_pool, max_active_trades=10):
        """Initialize with a database connection pool."""
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        # Initialize the Trade Manager
        self.trade_manager = TradeManager(db_pool, max_active_trades)

    async def get_variable_weights(self):
        """
        Returns hardcoded weights for various bot metrics.
        
        This implementation provides a reliable baseline for bot ranking without
        depending on database entries. The weights are chosen based on their importance
        to overall trading performance.
        
        Returns:
            Dictionary of variable names and their corresponding weights
        """
        # Hardcoded weights for different metrics
        weights = {
            # Performance periods - more recent periods have higher weight
            'one_hour_performance': 15.0,   # Most recent performance is highly valued
            'two_hour_performance': 10.0,
            'one_day_performance': 12.0,
            'one_week_performance': 8.0,
            'one_month_performance': 5.0,   # Long-term stability is still important
            
            # Core metrics - fundamental indicators of success
            'avg_win_rate': 12.0,          # Consistency in winning trades
            'profit_per_second': 10.0,     # Efficiency of profit generation
            'total_pnl': 8.0,              # Total profit is important
            
            # Trade statistics
            'profit_factor': 8.0,          # Ratio of gains to losses
            'avg_profit_per_trade': 6.0,   # Average profit per trade
            
            # Risk metrics - lower is better
            'avg_drawdown': -5.0,          # Negative weight as lower drawdown is better
            'max_drawdown': -7.0,          # Negative weight as lower max drawdown is better
            'sharpe_ratio': 8.0,           # Risk-adjusted return measure
            
            # Model scores - algorithmic effectiveness
            'price_model_score': 9.0,      # Effectiveness of price prediction
            'volume_model_score': 7.0,     # Effectiveness of volume analysis
            'price_wall_score': 6.0,       # Effectiveness of support/resistance analysis
            
            # Win streaks - psychological and momentum indicators
            'win_streak_2': 3.0,
            'win_streak_3': 4.0,
            'win_streak_4': 5.0,
            'win_streak_5': 6.0,
        }
        
        return weights

    async def fetch_bot_metrics(self):
        """Fetch the latest metrics for all bots."""
        try:
            async with self.db_pool.acquire() as connection:
                # Get the latest metrics for each bot
                metrics = await connection.fetch("""
                    SELECT DISTINCT ON (bot_id) 
                        bot_id, 
                        ticker,
                        algo_id,
                        one_hour_performance,
                        two_hour_performance,
                        one_day_performance,
                        one_week_performance,
                        one_month_performance,
                        avg_win_rate,
                        avg_drawdown,
                        max_drawdown,
                        profit_per_second,
                        total_pnl,
                        total_trades,
                        avg_profit_per_trade,
                        profit_factor,
                        sharpe_ratio,
                        price_model_score,
                        volume_model_score,
                        price_wall_score,
                        win_streak_2,
                        win_streak_3,
                        win_streak_4,
                        win_streak_5,
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
        and the hardcoded variable weights.
        
        Args:
            bot_metrics: Dictionary containing the bot's metrics
            weights: Dictionary of metric names and their weights
            
        Returns:
            Float value representing the bot's rank score
        """
        try:
            # Start with zero score
            score = Decimal('0.0')
            
            # List of all metrics that should be processed
            metrics_to_process = [
                # Performance periods
                'one_hour_performance', 'two_hour_performance', 'one_day_performance',
                'one_week_performance', 'one_month_performance',
                
                # Core metrics
                'avg_win_rate', 'profit_per_second', 'total_pnl',
                
                # Trade statistics
                'profit_factor', 'avg_profit_per_trade', 'total_trades',
                
                # Risk metrics
                'avg_drawdown', 'max_drawdown', 'sharpe_ratio',
                
                # Model scores
                'price_model_score', 'volume_model_score', 'price_wall_score',
            ]
            
            # Add win streak metrics
            for i in range(2, 6):  # Win streaks 2 through 5
                metrics_to_process.append(f'win_streak_{i}')
                
            # Process each metric
            for metric in metrics_to_process:
                if metric in weights and metric in bot_metrics and bot_metrics[metric] is not None:
                    weight = Decimal(str(weights[metric]))
                    value = Decimal(str(bot_metrics[metric]))
                    
                    # Handle metrics where lower is better (negative weights)
                    if weight < 0:
                        # Convert negative weights to positive and invert the value
                        # For metrics like drawdown where lower is better
                        weight = abs(weight)
                        
                        # Scale the value inversely (100 - value) so that lower values get higher scores
                        # This works for percentage-based metrics; adjust scaling as needed
                        if metric in ['avg_drawdown', 'max_drawdown']:
                            # Cap at 100 to prevent negative scores
                            capped_value = min(value, Decimal('100.0'))
                            adjusted_value = Decimal('100.0') - capped_value
                            score += weight * adjusted_value
                        else:
                            # For other metrics where lower is better but not percentage-based
                            # Use an inverse relationship
                            if value > Decimal('0.0'):
                                score += weight * (Decimal('1.0') / value)
                            else:
                                # Handle zero case to avoid division by zero
                                score += weight * Decimal('100.0')  # Max score for zero drawdown
                    else:
                        # For regular metrics where higher is better
                        score += weight * value
            
            # Apply normalization factor to keep scores in a reasonable range
            # Adjust this denominator based on the range of your weights and values
            normalization_factor = Decimal('100.0')
            score = score / normalization_factor
            
            return float(score)
        except Exception as e:
            self.logger.error(f"Error calculating bot rank: {e}")
            return 0.0

    async def rank_bots(self):
        """
        Rank all bots based on their metrics and hardcoded weights.
        Returns a list of bots sorted by rank (highest first).
        
        This implementation uses predefined weights to calculate a comprehensive 
        score for each bot based on its performance metrics.
        
        Returns:
            List of dictionaries, each containing bot information with rank scores.
        """
        try:
            # Get hardcoded weights
            weights = await self.get_variable_weights()
            
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
            
            # Assign rank numbers based on sorted order
            for i, bot in enumerate(ranked_bots):
                bot['rank'] = i + 1
            
            # Update rankings in database
            await self._update_bot_rankings(ranked_bots)
            
            return ranked_bots
        except Exception as e:
            self.logger.error(f"Error ranking bots: {e}")
            return []

    async def _update_bot_rankings(self, ranked_bots):
        """
        Update the bot_rankings table with the latest rankings.
        
        This method stores the rank score for each bot 
        in the database for record-keeping and decision-making.
        
        Args:
            ranked_bots: List of bot dictionaries with rank_score and rank values
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Check if table exists, create if not
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS bot_rankings (
                        ranking_id SERIAL PRIMARY KEY,
                        bot_id INTEGER NOT NULL,
                        rank_score DECIMAL(10,2) NOT NULL,
                        rank INTEGER NOT NULL,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT true,
                        UNIQUE(bot_id)
                    )
                """)
                
                # Update rankings for each bot
                for bot in ranked_bots:
                    await connection.execute("""
                        INSERT INTO bot_rankings (bot_id, rank_score, rank, timestamp)
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (bot_id) 
                        DO UPDATE SET 
                            rank_score = $2,
                            rank = $3,
                            timestamp = NOW()
                    """, bot['bot_id'], bot['rank_score'], bot['rank'])
                    
                # Also update the bot_metrics table with the current rank
                for bot in ranked_bots:
                    await connection.execute("""
                        UPDATE bot_metrics
                        SET current_rank = $2
                        WHERE bot_id = $1 AND timestamp = (
                            SELECT MAX(timestamp) FROM bot_metrics WHERE bot_id = $1
                        )
                    """, bot['bot_id'], bot['rank'])
                    
        except Exception as e:
            self.logger.error(f"Error updating bot rankings: {e}")

    async def get_fund_allocation(self, total_funds, max_allocation_pct=10.0, min_allocation_pct=1.0):
        """
        Calculate fund allocation based on bot rankings.
        
        Args:
            total_funds: Total funds available for trading
            max_allocation_pct: Maximum percentage to allocate to a single bot (default: 10%)
            min_allocation_pct: Minimum percentage to allocate to a bot (default: 1%)
            
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
            
            # Filter to only active bots
            ranked_active_bots = [bot for bot in ranked_bots if bot['bot_id'] in active_bot_ids]
            
            if not ranked_active_bots:
                return []
            
            # Calculate allocations based on rank scores
            allocations = []
            total_score = sum(bot['rank_score'] for bot in ranked_active_bots)
            
            # If total_score is 0, use equal distribution
            if total_score == 0:
                equal_share = total_funds / len(ranked_active_bots)
                for bot in ranked_active_bots:
                    allocations.append({
                        'bot_id': bot['bot_id'],
                        'ticker': bot['ticker'],
                        'rank_score': bot['rank_score'],
                        'rank': bot['rank'],
                        'allocation_amount': equal_share,
                        'allocation_percentage': 100.0 / len(ranked_active_bots)
                    })
            else:
                # Proportional allocation based on rank score, with min and max limits
                for bot in ranked_active_bots:
                    # Calculate raw percentage based on score
                    raw_percentage = (bot['rank_score'] / total_score) * 100.0
                    
                    # Apply min/max constraints
                    allocation_percentage = max(min(raw_percentage, max_allocation_pct), min_allocation_pct)
                    allocation_amount = (allocation_percentage / 100.0) * total_funds
                    
                    allocations.append({
                        'bot_id': bot['bot_id'],
                        'ticker': bot['ticker'],
                        'rank_score': bot['rank_score'],
                        'rank': bot['rank'],
                        'allocation_amount': allocation_amount,
                        'allocation_percentage': allocation_percentage
                    })
                
                # Normalize allocations to ensure they sum to 100%
                total_allocated_pct = sum(alloc['allocation_percentage'] for alloc in allocations)
                
                if total_allocated_pct != 100.0:
                    scale_factor = 100.0 / total_allocated_pct
                    
                    for alloc in allocations:
                        alloc['allocation_percentage'] *= scale_factor
                        alloc['allocation_amount'] = (alloc['allocation_percentage'] / 100.0) * total_funds
            
            # Sort by allocation amount (descending)
            allocations.sort(key=lambda x: x['allocation_amount'], reverse=True)
            
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
                
                # If activating a bot, ensure bot rankings are updated
                if is_active:
                    await self.trade_manager.update_bot_activations()
                
                return True
        except Exception as e:
            self.logger.error(f"Error toggling bot active status: {e}")
            return False
            
    # New methods to integrate with TradeManager
    
    async def can_bot_trade(self, bot_id):
        """
        Check if a bot is allowed to trade based on current portfolio allocation.
        
        Args:
            bot_id: The ID of the bot requesting to trade
            
        Returns:
            bool: True if the bot can open a new trade, False otherwise
        """
        can_trade, _, _ = await self.trade_manager.can_open_new_trade(bot_id)
        return can_trade
        
    async def initiate_bot_trade(self, bot_id, ticker, entry_price, trade_direction, trade_size):
        """
        Initiate a trade for a bot using the dynamic allocation logic.
        
        Args:
            bot_id: The ID of the bot initiating the trade
            ticker: The stock ticker symbol
            entry_price: The price at which to enter the trade
            trade_direction: 'LONG' or 'SHORT'
            trade_size: The size of the trade in dollars
            
        Returns:
            dict: Status information about the trade initiation
        """
        return await self.trade_manager.initiate_trade(
            bot_id, ticker, entry_price, trade_direction, trade_size
        )
        
    async def complete_bot_trade(self, trade_id, exit_price):
        """
        Complete a trade with the given trade_id and exit_price.
        
        Args:
            trade_id: The ID of the trade to complete
            exit_price: The price at which to exit the trade
            
        Returns:
            dict: Status information about the trade completion
        """
        return await self.trade_manager.complete_trade(trade_id, exit_price)
    
    async def update_all_bot_activations(self):
        """
        Update all bot activations based on current trades.
        This should be called periodically to ensure proper activations.
        """
        return await self.trade_manager.update_bot_activations()
        
    async def get_trade_dashboard_data(self):
        """
        Get data for a dashboard showing active trades and bot statuses.
        
        Returns:
            dict: Dashboard data including active trades, bot activations, and portfolio usage
        """
        return await self.trade_manager.get_trade_dashboard_data()
