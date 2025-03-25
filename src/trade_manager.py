import logging
import asyncpg
from datetime import datetime

class TradeManager:
    """
    Manages the dynamic allocation of trade slots based on bot rankings.
    Implements the strategy of allowing up to a maximum number of concurrent
    trades, with higher-ranked bots able to replace trades from lower-ranked bots.
    """
    
    def __init__(self, db_pool, max_active_trades=10):
        """Initialize with a database connection pool and maximum allowed trades."""
        self.db_pool = db_pool
        self.max_active_trades = max_active_trades
        self.logger = logging.getLogger(__name__)
        
    async def get_active_trades(self):
        """
        Fetch all currently active trades ordered by bot rank score (descending).
        
        Returns:
            list: Active trades with bot ranking information
        """
        try:
            async with self.db_pool.acquire() as connection:
                active_trades = await connection.fetch("""
                    SELECT 
                        st.trade_id, 
                        st.bot_id, 
                        st.ticker, 
                        st.entry_time, 
                        st.entry_price,
                        st.trade_direction,
                        st.trade_size,
                        br.rank_score
                    FROM sim_bot_trades st
                    JOIN bot_rankings br ON st.bot_id = br.bot_id
                    WHERE st.trade_status = 'open'
                    ORDER BY br.rank_score DESC
                """)
                return active_trades
        except Exception as e:
            self.logger.error(f"Error getting active trades: {e}")
            return []
    
    async def get_portfolio_usage(self):
        """
        Calculate current portfolio usage as a percentage.
        
        Returns:
            float: Percentage of portfolio currently in use (0-100)
        """
        try:
            active_trades = await self.get_active_trades()
            return (len(active_trades) / self.max_active_trades) * 100
        except Exception as e:
            self.logger.error(f"Error calculating portfolio usage: {e}")
            return 0.0
    
    async def can_open_new_trade(self, bot_id):
        """
        Determine if a bot can open a new trade based on current portfolio allocation.
        
        Args:
            bot_id: The ID of the bot requesting to open a trade
            
        Returns:
            tuple: (can_open, needs_to_close_trade, lowest_ranked_trade)
        """
        try:
            # Get current active trades
            active_trades = await self.get_active_trades()
            
            # Get the bot's rank score
            async with self.db_pool.acquire() as connection:
                bot_rank = await connection.fetchval("""
                    SELECT rank_score FROM bot_rankings
                    WHERE bot_id = $1
                """, bot_id)
                
                if bot_rank is None:
                    self.logger.warning(f"Bot {bot_id} has no ranking record")
                    return (False, False, None)
            
            # If we have fewer than max trades, allow new trade
            if len(active_trades) < self.max_active_trades:
                self.logger.info(f"Portfolio has capacity: {len(active_trades)}/{self.max_active_trades} trades")
                return (True, False, None)
            
            # If we're at max trades, check if this bot outranks the lowest-ranked active trade
            lowest_ranked_trade = active_trades[-1]
            
            if bot_rank > lowest_ranked_trade['rank_score']:
                # This bot outranks the lowest active trade, so we can close that one
                self.logger.info(f"Bot {bot_id} (score: {bot_rank}) outranks lowest active bot {lowest_ranked_trade['bot_id']} (score: {lowest_ranked_trade['rank_score']})")
                return (True, True, lowest_ranked_trade)
            
            # Otherwise, this bot doesn't get to trade now
            self.logger.info(f"Bot {bot_id} (score: {bot_rank}) cannot trade - portfolio full and ranked too low")
            return (False, False, None)
            
        except Exception as e:
            self.logger.error(f"Error checking if bot {bot_id} can open trade: {e}")
            return (False, False, None)
    
    async def close_lowest_ranked_trade(self):
        """
        Close the lowest-ranked active trade to make room for a higher-ranked trade.
        
        Returns:
            dict: Information about the closed trade, or None if failed
        """
        try:
            active_trades = await self.get_active_trades()
            
            if not active_trades:
                self.logger.warning("No active trades to close")
                return None
                
            # Get the lowest-ranked trade
            lowest_ranked_trade = active_trades[-1]
            trade_id = lowest_ranked_trade['trade_id']
            bot_id = lowest_ranked_trade['bot_id']
            
            # Close the trade
            async with self.db_pool.acquire() as connection:
                # Get current market price for the ticker
                ticker = lowest_ranked_trade['ticker']
                current_price = await connection.fetchval("""
                    SELECT price FROM tick_data
                    WHERE ticker = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, ticker)
                
                if current_price is None:
                    self.logger.error(f"Could not get current price for {ticker}")
                    return None
                
                # Calculate PnL based on trade direction
                if lowest_ranked_trade['trade_direction'] == 'LONG':
                    pnl = current_price - lowest_ranked_trade['entry_price']
                else:  # SHORT
                    pnl = lowest_ranked_trade['entry_price'] - current_price
                
                # Update the trade to closed status
                await connection.execute("""
                    UPDATE sim_bot_trades
                    SET 
                        trade_status = 'closed',
                        exit_time = NOW(),
                        exit_price = $1,
                        trade_pnl = $2
                    WHERE trade_id = $3
                """, current_price, pnl, trade_id)
                
                self.logger.info(f"Force-closed trade {trade_id} for Bot {bot_id} with PnL: ${pnl:.2f}")
                
                return {
                    'trade_id': trade_id,
                    'bot_id': bot_id,
                    'exit_price': current_price,
                    'pnl': pnl
                }
        except Exception as e:
            self.logger.error(f"Error closing lowest-ranked trade: {e}")
            return None
    
    async def update_bot_activations(self):
        """
        Update bot active status based on current trades.
        Deactivate all bots ranked lower than the lowest-ranked bot with an active trade
        when portfolio is fully allocated.
        """
        try:
            active_trades = await self.get_active_trades()
            
            # If no active trades or not at capacity, activate all bots
            if not active_trades or len(active_trades) < self.max_active_trades:
                async with self.db_pool.acquire() as connection:
                    await connection.execute("""
                        UPDATE bot_rankings
                        SET is_active = true
                    """)
                self.logger.info(f"Portfolio not at capacity ({len(active_trades)}/{self.max_active_trades}) - activated all bots")
                return
            
            # At capacity, deactivate bots ranked lower than the lowest active trade
            lowest_active_bot = active_trades[-1]
            lowest_rank_score = lowest_active_bot['rank_score']
            
            async with self.db_pool.acquire() as connection:
                # Activate all bots with rank_score >= lowest_active_bot's score
                await connection.execute("""
                    UPDATE bot_rankings
                    SET is_active = (rank_score >= $1)
                """, lowest_rank_score)
                
                # Get count of deactivated bots
                deactivated_count = await connection.fetchval("""
                    SELECT COUNT(*) FROM bot_rankings WHERE is_active = false
                """)
            
            self.logger.info(f"Portfolio at capacity - deactivated {deactivated_count} bots ranked lower than bot {lowest_active_bot['bot_id']} (score: {lowest_rank_score})")
            
        except Exception as e:
            self.logger.error(f"Error updating bot activations: {e}")
    
    async def initiate_trade(self, bot_id, ticker, entry_price, trade_direction, trade_size=None):
        """
        Initiate a new trade with the dynamic allocation logic using fixed trade size.
        
        All trades use exactly 10% of the total fund allocation ($2,000 out of $20,000).
        
        Args:
            bot_id: The ID of the bot initiating the trade
            ticker: The stock ticker symbol
            entry_price: The price at which to enter the trade
            trade_direction: 'LONG' or 'SHORT'
            trade_size: Optional - if not provided, uses fixed 10% of total funds ($2,000)
            
        Returns:
            dict: Status information about the trade initiation
        """
        # Set default trade size to fixed 10% of $20,000 total funds
        if trade_size is None:
            trade_size = 2000.0  # Fixed at 10% of $20,000
        try:
            # Check if this bot can open a trade
            can_trade, needs_to_close, lowest_trade = await self.can_open_new_trade(bot_id)
            
            if not can_trade:
                self.logger.info(f"Bot {bot_id} cannot trade: portfolio capacity reached and bot ranked too low")
                return {
                    'success': False,
                    'reason': 'Portfolio capacity reached and bot ranked too low'
                }
                
            # Prepare result object
            result = {
                'success': True,
                'closed_trade': None
            }
                
            # If needed, close the lowest-ranked trade
            if needs_to_close:
                closed_trade = await self.close_lowest_ranked_trade()
                if not closed_trade:
                    self.logger.error("Failed to close lowest-ranked trade")
                    return {
                        'success': False,
                        'reason': 'Failed to close lowest-ranked trade'
                    }
                
                result['closed_trade'] = closed_trade
                self.logger.info(f"Closed lowest-ranked trade {closed_trade['trade_id']} to make room for bot {bot_id}")
            
            # Open the new trade
            async with self.db_pool.acquire() as connection:
                trade_id = await connection.fetchval("""
                    INSERT INTO sim_bot_trades (
                        bot_id, ticker, entry_time, entry_price, 
                        trade_direction, trade_size, trade_status
                    )
                    VALUES ($1, $2, NOW(), $3, $4, $5, 'open')
                    RETURNING trade_id
                """, bot_id, ticker, entry_price, trade_direction, trade_size)
                
                self.logger.info(f"Opened new trade {trade_id} for bot {bot_id} ({ticker} {trade_direction})")
                
                result['trade_id'] = trade_id
            
            # Update bot activations based on new trade state
            await self.update_bot_activations()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error initiating trade for bot {bot_id}: {e}")
            return {
                'success': False,
                'reason': f'Exception: {str(e)}'
            }
    
    async def complete_trade(self, trade_id, exit_price):
        """
        Complete an existing trade.
        
        Args:
            trade_id: The ID of the trade to complete
            exit_price: The price at which to exit the trade
            
        Returns:
            dict: Status information about the trade completion
        """
        try:
            async with self.db_pool.acquire() as connection:
                # Get trade details
                trade = await connection.fetchrow("""
                    SELECT bot_id, ticker, entry_price, trade_direction, trade_size
                    FROM sim_bot_trades
                    WHERE trade_id = $1 AND trade_status IN ('open', 'pending_exit')
                """, trade_id)
                
                if not trade:
                    self.logger.error(f"Trade {trade_id} not found or not in open/pending_exit state")
                    return {
                        'success': False,
                        'reason': 'Trade not found or not in open/pending_exit state'
                    }
                    
                # Calculate PnL - convert all values to float to avoid type mismatches
                exit_price_float = float(exit_price)
                entry_price_float = float(trade['entry_price'])
                trade_size_float = float(trade['trade_size'])
                
                if trade['trade_direction'] == 'LONG':
                    pnl = (exit_price_float - entry_price_float) * (trade_size_float / entry_price_float)
                else:  # SHORT
                    pnl = (entry_price_float - exit_price_float) * (trade_size_float / entry_price_float)
                    
                # Update trade
                await connection.execute("""
                    UPDATE sim_bot_trades
                    SET 
                        trade_status = 'closed',
                        exit_time = NOW(),
                        exit_price = $1,
                        trade_pnl = $2
                    WHERE trade_id = $3
                """, exit_price, pnl, trade_id)
                
                # Get the trade status for better logging
                trade_status = await connection.fetchval("""
                    SELECT trade_status FROM sim_bot_trades WHERE trade_id = $1
                """, trade_id)
                
                self.logger.info(f"Completed trade {trade_id} with PnL: ${pnl:.2f} (previous status: {trade_status})")
            
            # Update bot activations since we've completed a trade
            await self.update_bot_activations()
            
            return {
                'success': True,
                'trade_id': trade_id,
                'pnl': pnl
            }
            
        except Exception as e:
            self.logger.error(f"Error completing trade {trade_id}: {e}")
            return {
                'success': False,
                'reason': f'Exception: {str(e)}'
            }
    
    async def get_trade_dashboard_data(self):
        """
        Get data for a dashboard showing active trades and bot statuses.
        
        Returns:
            dict: Dashboard data including active trades, bot activations, and portfolio usage
        """
        try:
            result = {
                'active_trades': [],
                'bot_statuses': [],
                'portfolio_usage': 0.0
            }
            
            # Get active trades with bot details
            async with self.db_pool.acquire() as connection:
                active_trades = await connection.fetch("""
                    SELECT 
                        st.trade_id, 
                        st.bot_id, 
                        st.ticker, 
                        st.entry_time, 
                        st.entry_price,
                        st.trade_direction,
                        st.trade_size,
                        br.rank_score,
                        br.is_active
                    FROM sim_bot_trades st
                    JOIN bot_rankings br ON st.bot_id = br.bot_id
                    WHERE st.trade_status = 'open'
                    ORDER BY br.rank_score DESC
                """)
                
                # Calculate portfolio usage
                result['active_trades'] = [dict(t) for t in active_trades]
                result['portfolio_usage'] = len(active_trades) / self.max_active_trades * 100
                
                # Get bot statuses
                bot_statuses = await connection.fetch("""
                    SELECT 
                        br.bot_id, 
                        bm.ticker, 
                        br.rank_score, 
                        br.is_active,
                        EXISTS (
                            SELECT 1 FROM sim_bot_trades st 
                            WHERE st.bot_id = br.bot_id AND st.trade_status = 'open'
                        ) as has_active_trade
                    FROM bot_rankings br
                    LEFT JOIN bot_metrics bm ON br.bot_id = bm.bot_id
                    ORDER BY br.rank_score DESC
                """)
                
                result['bot_statuses'] = [dict(s) for s in bot_statuses]
                
                # Add lowest active rank info
                if active_trades:
                    lowest_active = active_trades[-1]
                    result['lowest_active_rank_score'] = lowest_active['rank_score']
                    result['lowest_active_bot_id'] = lowest_active['bot_id']
                else:
                    result['lowest_active_rank_score'] = None
                    result['lowest_active_bot_id'] = None
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting trade dashboard data: {e}")
            return None
