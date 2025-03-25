"""
Base Trading Bot Class

This module provides a base class for trading bots that load their
algorithm configuration from YAML files and use algorithm-specific
Python modules for entry/exit logic.
"""

import asyncio
import logging
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import importlib
import sys

# Add current directory to path to help with imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from trade management system
try:
    # Try relative import first
    from bot_ranker import BotRanker
except ImportError:
    # Try with full path
    from src.bot_ranker import BotRanker

# Import IB API modules
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class IBClient(EWrapper, EClient):
    """Interactive Brokers client implementation for handling market data and order execution."""
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False

    def connectAck(self):
        self.connected = True
        logging.info("Connected to IB Gateway")

    def error(self, reqId, errorCode, errorString):
        logging.error(f"IB Error {errorCode}: {errorString}")


class BaseBot:
    """
    Base Trading Bot Class
    
    Loads algorithm configuration from YAML files and implements the trading logic
    while handling database connections, broker interactions, and trade management.
    """
    
    def __init__(self, db_pool, ib_client, algorithm_file):
        """
        Initialize the trading bot.
        
        Args:
            db_pool: Database connection pool
            ib_client: Interactive Brokers client
            algorithm_file: Path to YAML file containing the algorithm configuration
        """
        # Load algorithm configuration
        self.config = self.load_algorithm_config(algorithm_file)
        
        # Set up basic parameters from config
        self.bot_id = self.config.get('bot_id')
        self.ticker = self.config.get('ticker')
        self.algo_id = self.config.get('algo_id')
        self.trade_direction = self.config.get('trade_direction', 'LONG')
        self.position_size = float(self.config.get('position_size', 10000.0))
        self.trailing_stop_pct = float(self.config.get('trailing_stop_pct', 0.002))
        
        # Set up logger with bot ID
        self.logger = logging.getLogger(f"Bot-{self.bot_id}")
        
        # Store connections
        self.db_pool = db_pool
        self.ib_client = ib_client
        
        # Set up state variables
        self.position = None  # Will be set when in a position
        self.entry_price = None  # Price at which we entered a position
        self.current_trade_id = None  # ID of the current trade
        self.extreme_price = 0  # Highest price for LONG, lowest for SHORT
        
        # Initialize bot ranker for trade management
        self.bot_ranker = BotRanker(db_pool)
        
        # Load algorithm module
        self.algorithm = self.load_algorithm_module()
        
        self.logger.info(f"Initialized bot with algorithm: {self.config.get('name')}")
        
    def load_algorithm_config(self, algorithm_file) -> Dict[str, Any]:
        """
        Load and parse the algorithm configuration from YAML file.
        
        Args:
            algorithm_file: Path to YAML configuration file
            
        Returns:
            Dict containing the algorithm configuration
        """
        try:
            with open(algorithm_file, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            logging.error(f"Error loading algorithm config: {e}")
            # Return a minimal default configuration
            return {
                'name': 'Default Config',
                'bot_id': 0,
                'ticker': 'UNKNOWN',
                'algo_id': 0,
                'trade_direction': 'LONG',
                'position_size': 10000.0,
                'trailing_stop_pct': 0.002,
                'algorithm_module': None,
                'parameters': {}
            }
    
    def load_algorithm_module(self):
        """
        Load the algorithm-specific Python module.
        
        Returns:
            Algorithm instance or None if loading fails
        """
        module_path = self.config.get('algorithm_module')
        if not module_path:
            self.logger.error("No algorithm module specified in config")
            return None
            
        try:
            # Try to import the module directly
            try:
                module = importlib.import_module(module_path)
                self.logger.info(f"Successfully imported module {module_path}")
            except ImportError as e:
                self.logger.error(f"Error importing algorithm module {module_path}: {e}")
                self.logger.info(f"Trying alternative import paths...")
                
                # If this is specifying algorithms.X, try just importing X directly
                # This handles the case where algorithms/ is in the Python path
                if "." in module_path:
                    module_name = module_path.split(".")[-1]
                    try:
                        module = importlib.import_module(module_name)
                        self.logger.info(f"Successfully imported module {module_name}")
                    except ImportError:
                        # Try importing from parent directory
                        try:
                            module = importlib.import_module(f"..{module_path}", package="src")
                            self.logger.info(f"Successfully imported module ..{module_path}")
                        except ImportError as e2:
                            self.logger.error(f"All import attempts failed for {module_path}: {e2}")
                            return None
            
            # Get algorithm class name based on the module name
            algo_name = module_path.split('.')[-1]  # e.g. "momentum_algorithm"
            
            # Try different naming patterns for the algorithm class
            possible_class_names = [
                f"{algo_name.capitalize()}Algorithm",  # MomentumAlgorithm
                f"{algo_name.replace('_', '').capitalize()}Algorithm",  # MomentumAlgorithm
                f"{algo_name.split('_')[0].capitalize()}Algorithm"  # MomentumAlgorithm
            ]
            
            # Add the actual class names we found in our algorithm files
            if algo_name == "momentum_algorithm":
                possible_class_names.append("MomentumAlgorithm")
            elif algo_name == "breakout_algorithm":
                possible_class_names.append("BreakoutAlgorithm")
            elif algo_name == "mean_reversion_algorithm":
                possible_class_names.append("Mean_reversionAlgorithm")
            elif algo_name == "minute_momentum_algorithm":
                possible_class_names.append("Minute_momentumAlgorithm")
            elif algo_name == "price_pattern_algorithm":
                possible_class_names.append("Price_patternAlgorithm")
            elif algo_name == "support_resistance_algorithm":
                possible_class_names.append("Support_resistanceAlgorithm")
            elif algo_name == "volatility_breakout_algorithm":
                possible_class_names.append("Volatility_breakoutAlgorithm")
            elif algo_name == "volume_surge_algorithm":
                possible_class_names.append("Volume_surgeAlgorithm")
            
            # Try each possible class name
            for class_name in possible_class_names:
                if hasattr(module, class_name):
                    algorithm_class = getattr(module, class_name)
                    # Create instance with parameters from config
                    parameters = self.config.get('parameters', {})
                    algorithm = algorithm_class(self.trade_direction, parameters)
                    self.logger.info(f"Successfully loaded algorithm class {class_name}")
                    return algorithm
            
            # If we get here, we couldn't find a matching class
            self.logger.error(f"No matching algorithm class found in {module_path}. Tried: {possible_class_names}")
            return None
                
        except Exception as e:
            self.logger.error(f"Error initializing algorithm: {e}")
            return None
    
    async def get_latest_ticks(self, seconds=60) -> Optional[pd.DataFrame]:
        """
        Fetch the latest tick data for the specified ticker.
        
        Args:
            seconds: Number of seconds of historical data to fetch
            
        Returns:
            Pandas DataFrame with columns ['timestamp', 'price'] or None if error
        """
        try:
            lookback_seconds = self.config.get('parameters', {}).get('lookback_seconds', seconds)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    WITH latest_tick AS (
                        SELECT timestamp 
                        FROM tick_data 
                        WHERE ticker = $1
                        AND price > 0  -- Only consider valid prices
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    )
                    SELECT timestamp, price 
                    FROM tick_data 
                    WHERE ticker = $1
                    AND price > 0  -- Filter out invalid prices
                    AND timestamp >= (SELECT timestamp - INTERVAL '{lookback_seconds} seconds' FROM latest_tick)
                    ORDER BY timestamp DESC;
                """, self.ticker)

                # Convert the result into a Pandas DataFrame
                if not rows:
                    self.logger.warning(f"No tick data available for {self.ticker}")
                    return None
                    
                df = pd.DataFrame(rows, columns=['timestamp', 'price'])
                
                if len(df) > 0:
                    self.logger.debug(f"Fetched {len(df)} ticks, latest price: {df['price'].iloc[0]}")
                
                return df

        except Exception as e:
            self.logger.error(f"Error fetching tick data: {e}")
            return None
    
    async def analyze_entry_conditions(self, ticks_df) -> bool:
        """
        Analyze if price conditions meet entry criteria by delegating to algorithm module.
        
        Args:
            ticks_df: DataFrame containing tick data
            
        Returns:
            bool: True if entry conditions are met, False otherwise
        """
        if ticks_df is None or len(ticks_df) < 2:
            return False
            
        if self.algorithm is None:
            self.logger.error("No algorithm module loaded, cannot check entry conditions")
            return False
            
        # Delegate to algorithm module
        return self.algorithm.check_entry_conditions(ticks_df)
    
    def check_exit_conditions(self, current_price) -> bool:
        """
        Check if exit conditions are met by delegating to algorithm module.
        
        Args:
            current_price: Current price of the ticker
            
        Returns:
            bool: True if exit conditions are met, False otherwise
        """
        if self.position is None:
            return False
            
        if self.algorithm is None:
            self.logger.error("No algorithm module loaded, cannot check exit conditions")
            return False
            
        # Prepare position data for algorithm
        position_data = {
            'entry_price': self.entry_price,
            'extreme_price': self.extreme_price,
            'trailing_stop_pct': self.trailing_stop_pct
        }
        
        # Delegate to algorithm module
        exit_signal = self.algorithm.check_exit_conditions(current_price, position_data)
        
        # Update extreme price from algorithm's tracking
        self.extreme_price = position_data.get('extreme_price', self.extreme_price)
        
        return exit_signal
    
    async def execute_trade(self, action, price, timestamp):
        """
        Execute a trade order with trade management integration.
        
        Args:
            action: "BUY" or "SELL" action
            price: Current price for the trade
            timestamp: Timestamp of the trade signal
        """
        try:
            # Determine if this is an entry or exit based on action and trade direction
            is_entry = (action == "BUY" and self.trade_direction == "LONG") or \
                       (action == "SELL" and self.trade_direction == "SHORT")
            
            is_exit = (action == "SELL" and self.trade_direction == "LONG") or \
                      (action == "BUY" and self.trade_direction == "SHORT")
            
            if is_entry:
                # Check if this bot is allowed to trade
                can_trade = await self.bot_ranker.can_bot_trade(self.bot_id)
                
                if not can_trade:
                    self.logger.info(f"Cannot open trade: portfolio is full or bot {self.bot_id} is ranked too low")
                    return
                
                # Initiate the trade through the trade manager
                trade_result = await self.bot_ranker.initiate_bot_trade(
                    self.bot_id, 
                    self.ticker,
                    price, 
                    self.trade_direction, 
                    self.position_size
                )
                
                if not trade_result['success']:
                    self.logger.error(f"Failed to initiate trade: {trade_result.get('reason', 'Unknown error')}")
                    return
                
                # If a lower-ranked trade was closed to make room for this one, log it
                if 'closed_trade' in trade_result and trade_result['closed_trade']:
                    closed_trade = trade_result['closed_trade']
                    self.logger.info(f"Closed lower-ranked trade {closed_trade['trade_id']} from bot {closed_trade['bot_id']} to make room for this trade")
                
                # Update bot state
                self.position = 1
                self.entry_price = price
                self.current_trade_id = trade_result['trade_id']
                
                # Set initial extreme price (highest for LONG, lowest for SHORT)
                if self.trade_direction == 'LONG':
                    self.extreme_price = price
                else:  # SHORT
                    self.extreme_price = price
                
                self.logger.info(f"{action} executed at {price:.4f} to open {self.trade_direction} position")
            
            elif is_exit and self.current_trade_id:
                # Calculate PnL for logging
                if self.trade_direction == 'LONG':
                    pnl_pct = (price - self.entry_price) / self.entry_price * 100
                else:  # SHORT
                    pnl_pct = (self.entry_price - price) / self.entry_price * 100
                
                self.logger.info(f"{action} signal at {price:.4f}. PnL: {pnl_pct:.2f}%")
                
                # Log exit signal in the database
                await self.log_exit_signal(price, timestamp)
                
                # Complete the trade through the trade manager
                try:
                    trade_result = await self.bot_ranker.complete_bot_trade(
                        self.current_trade_id,
                        price
                    )
                    
                    if not trade_result['success']:
                        self.logger.error(f"Failed to complete trade: {trade_result.get('reason', 'Unknown error')}")
                        # Don't reset internal state if the trade wasn't completed successfully
                        # This way the bot will try again on the next tick
                        self.logger.warning(f"Trade {self.current_trade_id} still pending completion")
                        return
                        
                    # Only reset position state if the trade was completed successfully
                    self.position = None
                    self.entry_price = None
                    self.extreme_price = 0
                    self.current_trade_id = None
                    
                    self.logger.info(f"Position closed at {price:.4f}")
                except Exception as e:
                    self.logger.error(f"Exception completing trade: {e}")
                    self.logger.warning(f"Trade {self.current_trade_id} still pending completion")
                    # Don't reset internal state so the bot will try again

        except Exception as e:
            self.logger.error(f"Error executing {action} trade: {e}")
            raise
    
    async def log_exit_signal(self, price, timestamp):
        """Log when exit conditions are first met."""
        try:
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sim_bot_trades
                    SET exit_trigger_price = $1,
                        exit_trigger_time = $2,
                        trade_status = 'pending_exit'
                    WHERE trade_id = $3
                """, price, timestamp, self.current_trade_id)
        except Exception as e:
            self.logger.error(f"Error in log_exit_signal: {e}")
            raise
    
    async def run(self):
        """Main bot loop."""
        self.logger.info(f"Starting {self.config.get('name')} bot (ID: {self.bot_id})...")

        # Establish connection with Interactive Brokers if not already connected
        if not self.ib_client.connected:
            self.ib_client.connect('127.0.0.1', 4002, self.bot_id)  # Use bot_id as client ID
            
            # Wait until connection is confirmed
            while not self.ib_client.connected:
                await asyncio.sleep(0.1)

            self.logger.info("Connected to Interactive Brokers")

        while True:
            try:
                # Fetch latest tick data
                ticks_df = await self.get_latest_ticks()
                if ticks_df is None or len(ticks_df) == 0:
                    self.logger.info("No tick data available")
                    await asyncio.sleep(1)
                    continue

                current_price = float(ticks_df['price'].iloc[0])
                current_timestamp = ticks_df['timestamp'].iloc[0]
                
                # If already in a position, check exit conditions
                if self.position is not None:
                    if self.check_exit_conditions(current_price):
                        # Determine action based on trade direction
                        action = "SELL" if self.trade_direction == "LONG" else "BUY"
                        await self.execute_trade(action, current_price, current_timestamp)
                
                # If not in a position, check entry conditions
                else:
                    if await self.analyze_entry_conditions(ticks_df):
                        # Determine action based on trade direction
                        action = "BUY" if self.trade_direction == "LONG" else "SELL"
                        await self.execute_trade(action, current_price, current_timestamp)

                # Pause before next cycle
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    async def process_tick(self, ticker, price, timestamp):
        """
        Process each new tick of market data.
        
        Args:
            ticker: The ticker symbol
            price: The current price
            timestamp: The timestamp of the tick
        """
        # Skip if not our target symbol
        if ticker != self.ticker:
            return
            
        # Log the tick for debugging
        self.logger.debug(f"Processing tick: {ticker} @ ${price:.2f} ({timestamp})")
        
        # Skip if price is invalid (negative or zero)
        if price <= 0:
            self.logger.warning(f"Skipping invalid price: {price}")
            return
            
        # Get latest ticks for analysis
        ticks_df = await self.get_latest_ticks()
        
        # Check for exit if in a position
        if self.position:
            # Update extreme price for trailing stop
            if self.trade_direction == 'LONG' and price > self.extreme_price:
                self.extreme_price = price
            elif self.trade_direction == 'SHORT' and price < self.extreme_price:
                self.extreme_price = price
                
            # Check exit conditions
            if self.algorithm and self.algorithm.check_exit_conditions(price, {
                'entry_price': self.entry_price,
                'extreme_price': self.extreme_price,
                'direction': self.trade_direction
            }):
                action = "SELL" if self.trade_direction == "LONG" else "BUY"
                await self.execute_trade(action, price, timestamp)
        
        # Check for entry if not in a position
        elif ticks_df is not None and len(ticks_df) > 0:
            if await self.analyze_entry_conditions(ticks_df):
                action = "BUY" if self.trade_direction == "LONG" else "SELL"
                await self.execute_trade(action, price, timestamp)


class BotFactory:
    """
    Factory class for creating and managing trading bots
    """
    
    def __init__(self, db_pool, algorithm_dir="algorithms"):
        """
        Initialize the bot factory.
        
        Args:
            db_pool: Database connection pool
            algorithm_dir: Directory containing YAML algorithm files
        """
        self.db_pool = db_pool
        self.algorithm_dir = algorithm_dir
        self.logger = logging.getLogger("BotFactory")
        self.ib_client = IBClient()  # Single IB client for all bots
        self.bots = {}  # Dictionary of active bots
        self.last_valid_prices = {}  # Dictionary to store last valid price for each ticker
    
    def get_available_algorithms(self):
        """
        Get a list of available algorithm files.
        
        Returns:
            List of algorithm file paths
        """
        try:
            if not os.path.exists(self.algorithm_dir):
                self.logger.warning(f"Algorithm directory '{self.algorithm_dir}' does not exist")
                return []
                
            return [os.path.join(self.algorithm_dir, file) 
                    for file in os.listdir(self.algorithm_dir) 
                    if file.endswith('.yaml') or file.endswith('.yml')]
        except Exception as e:
            self.logger.error(f"Error getting available algorithms: {e}")
            return []
    
    async def create_bot(self, algorithm_file):
        """
        Create a new trading bot instance using the bot_id from the YAML file.
        
        Args:
            algorithm_file: Path to YAML algorithm file
            
        Returns:
            BaseBot instance
        """
        try:
            # Create a new bot instance
            bot = BaseBot(self.db_pool, self.ib_client, algorithm_file)
            
            # Store the bot instance using bot_id from config
            bot_id = bot.bot_id
            if bot_id is None:
                self.logger.error(f"Bot ID not specified in {algorithm_file}")
                return None
                
            self.bots[bot_id] = bot
            
            self.logger.info(f"Created bot {bot_id} with algorithm: {bot.config.get('name')}")
            
            return bot
        except Exception as e:
            self.logger.error(f"Error creating bot: {e}")
            raise
    
    async def start_bot(self, bot_id):
        """
        Start a trading bot.
        
        Args:
            bot_id: ID of the bot to start
            
        Returns:
            bool: True if the bot was started successfully, False otherwise
        """
        try:
            if bot_id not in self.bots:
                self.logger.error(f"Bot {bot_id} does not exist")
                return False
                
            # Start the bot in a background task
            bot = self.bots[bot_id]
            asyncio.create_task(bot.run())
            
            self.logger.info(f"Started bot {bot_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error starting bot {bot_id}: {e}")
            return False
    
    async def stop_bot(self, bot_id):
        """
        Stop a trading bot.
        
        Args:
            bot_id: ID of the bot to stop
            
        Returns:
            bool: True if the bot was stopped successfully, False otherwise
        """
        # Note: This is a stub - actual implementation would require a way to cancel the bot's task
        self.logger.info(f"Stopping bot {bot_id} is not yet implemented")
        return False
    
    async def start_all_bots(self):
        """
        Start all available bots.
        
        Returns:
            dict: Dictionary mapping bot_id to success/failure
        """
        results = {}
        for bot_id in self.bots:
            results[bot_id] = await self.start_bot(bot_id)
        return results

    async def store_tick_data(self, ticker: str, price: float, volume: int, timestamp: datetime):
        if not self.db_pool:
            self.logger.error("Database pool not initialized")
            return

        try:
            # If price is invalid, use last valid price if available
            if price <= 0 and ticker in self.last_valid_prices:
                self.logger.warning(f"Replacing invalid price {price} with last valid price {self.last_valid_prices[ticker]} for {ticker}")
                price = self.last_valid_prices[ticker]
            elif price > 0:
                # Store valid price for future reference
                self.last_valid_prices[ticker] = price
            
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO tick_data (ticker, price, volume, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', ticker, price, volume, timestamp)
        except Exception as e:
            self.logger.error(f"Failed to store tick data: {e}")
