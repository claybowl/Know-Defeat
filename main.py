import logging
from pathlib import Path
from src.data.collector import IBDataCollector
from src.trading.session import TradingSession
from src.analysis.probability_engine import ProbabilityEngine
from src.monitoring.system_monitor import SystemMonitor
from src.data.database import DatabaseManager
import yaml
import time

class TradingSystem:
    def __init__(self):
        # Load configuration
        self.config = self.load_config()
        
        # Initialize components
        self.setup_logging()
        self.db = DatabaseManager(self.config['database'])
        self.collector = IBDataCollector(self.config['interactive_brokers'])
        self.session = TradingSession(self.db, self.config['trading'])
        self.prob_engine = ProbabilityEngine(self.db)
        self.monitor = SystemMonitor(self.db)
        
    def load_config(self):
        config_path = Path('config/settings.yaml')
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def setup_logging(self):
        logging.basicConfig(
            filename='logs/trading.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def startup(self):
        """System startup sequence"""
        try:
            # Connect to database
            self.db.connect()
            
            # Connect to Interactive Brokers
            self.collector.connect()
            
            # Initialize trading session
            self.session.initialize()
            
            # Start system monitoring
            self.monitor.start()
            
            logging.info("Trading system successfully started")
            
        except Exception as e:
            logging.error(f"Startup failed: {str(e)}")
            self.shutdown()
            
    def run(self):
        """Main trading loop"""
        while self.monitor.is_market_open():
            try:
                # Update probability engine rankings
                self.prob_engine.update_rankings()
                
                # Process any pending trades
                self.session.process_trades()
                
                # Health check
                self.monitor.check_system_health()
                
                time.sleep(1)  # Prevent excessive CPU usage
                
            except Exception as e:
                logging.error(f"Runtime error: {str(e)}")
                if self.monitor.should_emergency_stop():
                    self.shutdown()
                    break
                    
    def shutdown(self):
        """Clean system shutdown"""
        try:
            self.session.close_all_positions()
            self.collector.disconnect()
            self.db.disconnect()
            self.monitor.stop()
            logging.info("Trading system shutdown complete")
        except Exception as e:
            logging.critical(f"Shutdown error: {str(e)}")

if __name__ == "__main__":
    system = TradingSystem()
    system.startup()
    system.run()
    