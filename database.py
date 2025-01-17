import psycopg2
from psycopg2.extras import DictCursor
import logging
from datetime import datetime

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.cur = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.config['dbname'],
                user=self.config['user'],
                password=self.config['password'],
                host=self.config['host'],
                port=self.config['port']
            )
            self.cur = self.conn.cursor(cursor_factory=DictCursor)
            logging.info("Database connection established")
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            raise
            
    def store_tick_data(self, tick):
        """Store market data tick"""
        sql = """
            INSERT INTO tick_data (timestamp, symbol, price, volume, bid, ask)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cur.execute(sql, (
            tick['timestamp'],
            tick['symbol'],
            tick['price'],
            tick['volume'],
            tick['bid'],
            tick['ask']
        ))
        self.conn.commit()
        
    def update_bot_metrics(self, bot_id, metrics):
        """Update bot performance metrics"""
        sql = """
            INSERT INTO bot_metrics (
                bot_id, symbol, algorithm_id, timeframe,
                win_rate, profit_factor, sharpe_ratio, last_updated
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bot_id, timeframe) DO UPDATE
            SET win_rate = EXCLUDED.win_rate,
                profit_factor = EXCLUDED.profit_factor,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                last_updated = EXCLUDED.last_updated
        """
        self.cur.execute(sql, (
            bot_id,
            metrics['symbol'],
            metrics['algorithm_id'],
            metrics['timeframe'],
            metrics['win_rate'],
            metrics['profit_factor'],
            metrics['sharpe_ratio'],
            datetime.now()
        ))
        self.conn.commit()

    def get_bot_rankings(self):
        """Get current bot rankings"""
        sql = """
            SELECT b.bot_id, b.symbol, b.current_rank, 
                   m.win_rate, m.profit_factor, m.sharpe_ratio
            FROM bots b
            JOIN bot_metrics m ON b.bot_id = m.bot_id
            WHERE b.is_active = true
            ORDER BY b.current_rank DESC
        """
        self.cur.execute(sql)
        return self.cur.fetchall()

    def disconnect(self):
        """Close database connection"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")
