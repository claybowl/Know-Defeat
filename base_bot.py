from abc import ABC, abstractmethod
import logging

class BaseBot(ABC):
    def __init__(self, symbol, config):
        self.symbol = symbol
        self.config = config
        self.active = False
        self.current_position = None
        self.required_funds = config.get('required_funds', 0)
        
    @abstractmethod
    def analyze_tick(self, tick_data):
        """Analyze new tick data and generate trading signals"""
        pass
        
    @abstractmethod
    def calculate_metrics(self):
        """Calculate performance metrics for this bot"""
        pass
        
    def activate(self):
        """Activate the bot for trading"""
        self.active = True
        logging.info(f"Bot activated for {self.symbol}")
        
    def deactivate(self):
        """Deactivate the bot"""
        self.active = False
        logging.info(f"Bot deactivated for {self.symbol}")
        
    def get_required_funds(self):
        """Get funds required for this bot to trade"""
        return self.required_funds
        
    def update_position(self, position):
        """Update current position information"""
        self.current_position = position
