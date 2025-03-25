"""
Trade Monitor - Advanced Real-time Trading System Monitor

Run this alongside your trading system to monitor trades, rankings, and metrics
with attention-grabbing colorful output.

Features:
- Highlights trade entries and exits with bold, colorful text
- Periodically shows active trade summaries
- Displays bot rankings and changes
- Monitors metric changes
- Uses colors and formatting to make important events stand out

Usage:
    python scripts/trade_monitor.py
"""

import asyncio
import asyncpg
import sys
import os
import time
import logging
from datetime import datetime, timedelta
from tabulate import tabulate
from decimal import Decimal
import json
import signal

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ANSI color codes for colorful terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright/Bold colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

# Configure logging with colorful formatter
class ColorfulFormatter(logging.Formatter):
    """Custom formatter for colorful log output"""
    
    def format(self, record):
        log_message = super().format(record)
        
        # Add colors based on log level
        if record.levelno == logging.INFO:
            return f"{Colors.WHITE}{log_message}{Colors.RESET}"
        elif record.levelno == logging.WARNING:
            return f"{Colors.YELLOW}{log_message}{Colors.RESET}"
        elif record.levelno == logging.ERROR:
            return f"{Colors.RED}{log_message}{Colors.RESET}"
        elif record.levelno == logging.CRITICAL:
            return f"{Colors.BG_RED}{Colors.WHITE}{log_message}{Colors.RESET}"
        elif record.levelno == logging.DEBUG:
            return f"{Colors.BRIGHT_BLACK}{log_message}{Colors.RESET}"
        
        return log_message

# Configure logger
logger = logging.getLogger("TradeMonitor")
logger.setLevel(logging.INFO)

# Console handler with our custom formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorfulFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Also log to a file for record keeping
file_handler = logging.FileHandler("logs/trade_monitor.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Database connection parameters
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

class TradeMonitor:
    """Monitor trades, rankings, and metrics with colorful output"""
    
    def __init__(self):
        self.conn = None
        self.running = True
        self.last_trade_id = 0
        self.last_ranking_check = time.time()
        self.last_active_trade_check = time.time()
        self.last_metrics_check = time.time()
        self.rankings_cache = {}  # Store previous rankings for comparison
        self.metrics_cache = {}   # Store previous metrics for comparison
        self.alert_history = {}   # Prevent duplicate alerts
        
        # Register signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("Shutdown signal received. Closing connections...")
        self.running = False
    
    async def connect(self):
        """Connect to the database"""
        try:
            self.conn = await asyncpg.connect(**DB_CONFIG)
            logger.info(f"{Colors.BOLD}{Colors.GREEN}Connected to database{Colors.RESET}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")
    
    def format_trade_entry(self, trade, bot_name):
        """Format a trade entry with attention-grabbing styling"""
        direction = trade['trade_direction']
        direction_color = Colors.GREEN if direction == 'LONG' else Colors.RED
        
        entry = f"""
{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE} !!! NEW TRADE ALERT !!! {Colors.RESET}
{Colors.BOLD}Bot #{trade['bot_id']} ({bot_name}) entered {direction_color}{direction}{Colors.RESET} position
{Colors.BOLD}Ticker: {Colors.YELLOW}{trade['ticker']}{Colors.RESET}
{Colors.BOLD}Entry Price: {Colors.CYAN}${float(trade['entry_price']):.2f}{Colors.RESET}
{Colors.BOLD}Size: {Colors.MAGENTA}${float(trade['trade_size']):.2f}{Colors.RESET}
{Colors.BOLD}Trade ID: {trade['trade_id']}{Colors.RESET}
{Colors.BOLD}Time: {trade['entry_time']}{Colors.RESET}
"""
        return entry
    
    def format_trade_exit(self, trade, bot_name, pnl, pnl_percent):
        """Format a trade exit with attention-grabbing styling"""
        direction = trade['trade_direction']
        direction_color = Colors.GREEN if direction == 'LONG' else Colors.RED
        
        # Color P&L based on whether it's positive or negative
        pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
        
        exit = f"""
{Colors.BOLD}{Colors.BG_YELLOW}{Colors.BLACK} !!! TRADE EXIT ALERT !!! {Colors.RESET}
{Colors.BOLD}Bot #{trade['bot_id']} ({bot_name}) exited {direction_color}{direction}{Colors.RESET} position
{Colors.BOLD}Ticker: {Colors.YELLOW}{trade['ticker']}{Colors.RESET}
{Colors.BOLD}Entry Price: {Colors.CYAN}${float(trade['entry_price']):.2f}{Colors.RESET}
{Colors.BOLD}Exit Price: {Colors.CYAN}${float(trade['exit_price']):.2f}{Colors.RESET}
{Colors.BOLD}P&L: {pnl_color}${pnl:.2f} ({pnl_percent:.2f}%){Colors.RESET}
{Colors.BOLD}Trade ID: {trade['trade_id']}{Colors.RESET}
{Colors.BOLD}Duration: {(trade['exit_time'] - trade['entry_time']).total_seconds():.1f} seconds{Colors.RESET}
"""
        return exit
    
    async def get_bot_name(self, bot_id):
        """Get bot name from database"""
        try:
            name = await self.conn.fetchval(
                "SELECT name FROM sim_bots WHERE bot_id = $1", 
                bot_id
            )
            return name or f"Bot-{bot_id}"
        except Exception as e:
            logger.error(f"Error fetching bot name: {e}")
            return f"Bot-{bot_id}"
    
    async def check_new_trades(self):
        """Check for new trade entries and exits"""
        try:
            # Check for new trade entries
            new_entries = await self.conn.fetch("""
                SELECT trade_id, bot_id, ticker, entry_price, entry_time, 
                       trade_direction, trade_size
                FROM sim_bot_trades
                WHERE trade_id > $1 AND trade_status = 'open'
                ORDER BY trade_id ASC
            """, self.last_trade_id)
            
            for trade in new_entries:
                bot_name = await self.get_bot_name(trade['bot_id'])
                entry_alert = self.format_trade_entry(trade, bot_name)
                logger.info(entry_alert)
                self.last_trade_id = max(self.last_trade_id, trade['trade_id'])
            
            # Check for recent trade exits - Calculate pnl_percent if it doesn't exist in the schema
            recent_exits = await self.conn.fetch("""
                SELECT trade_id, bot_id, ticker, entry_price, exit_price, 
                       entry_time, exit_time, trade_direction, trade_size, 
                       trade_pnl,
                       CASE
                           WHEN trade_direction = 'LONG' THEN ((exit_price / entry_price) - 1) * 100
                           WHEN trade_direction = 'SHORT' THEN ((entry_price / exit_price) - 1) * 100
                           ELSE 0
                       END AS pnl_percent
                FROM sim_bot_trades
                WHERE trade_status = 'closed' 
                AND exit_time > NOW() - INTERVAL '1 minute'
                ORDER BY exit_time DESC
            """)
            
            for trade in recent_exits:
                # Skip if we've already alerted for this trade exit
                if trade['trade_id'] in self.alert_history.get('exits', set()):
                    continue
                
                # Get the P&L values, defaulting to 0 if NULL
                pnl = float(trade['trade_pnl'] or 0)
                pnl_percent = float(trade['pnl_percent'] or 0)
                
                bot_name = await self.get_bot_name(trade['bot_id'])
                exit_alert = self.format_trade_exit(trade, bot_name, pnl, pnl_percent)
                logger.info(exit_alert)
                
                # Record that we've alerted for this trade
                if 'exits' not in self.alert_history:
                    self.alert_history['exits'] = set()
                self.alert_history['exits'].add(trade['trade_id'])
                
                # Keep alert history manageable
                if len(self.alert_history.get('exits', set())) > 100:
                    self.alert_history['exits'] = set(list(self.alert_history['exits'])[-50:])
            
            # Check for trades stuck in pending_exit status
            pending_trades = await self.conn.fetch("""
                SELECT trade_id, bot_id, ticker, entry_price, exit_trigger_price, 
                       trade_direction, trade_size, exit_trigger_time
                FROM sim_bot_trades
                WHERE trade_status = 'pending_exit'
                AND exit_trigger_time < NOW() - INTERVAL '10 seconds'
            """)
            
            if pending_trades:
                logger.warning(f"\n{Colors.BOLD}{Colors.BG_RED}{Colors.WHITE} !!! STUCK TRADES ALERT !!! {Colors.RESET}")
                logger.warning(f"{Colors.YELLOW}Found {len(pending_trades)} trades stuck in pending_exit status{Colors.RESET}")
                
                # Print details of stuck trades
                table_data = []
                for trade in pending_trades:
                    stuck_duration = datetime.now() - trade['exit_trigger_time']
                    duration_str = str(stuck_duration).split('.')[0]  # Remove microseconds
                    
                    bot_name = await self.get_bot_name(trade['bot_id'])
                    
                    table_data.append([
                        trade['trade_id'],
                        f"{trade['bot_id']} ({bot_name})",
                        trade['ticker'],
                        trade['trade_direction'],
                        f"${float(trade['entry_price']):.2f}",
                        f"${float(trade['exit_trigger_price']):.2f}",
                        duration_str
                    ])
                
                # Print table of stuck trades
                headers = ["ID", "Bot", "Ticker", "Direction", "Entry", "Exit Trigger", "Stuck For"]
                table = tabulate(table_data, headers=headers, tablefmt="pretty")
                logger.warning(f"{table}")
                
                # Prompt for action
                logger.warning(f"{Colors.YELLOW}Consider running the pending_trade_fixer.py script to fix these trades{Colors.RESET}")
            
        except Exception as e:
            logger.error(f"Error checking for new trades: {e}")
    
    async def show_active_trades(self):
        """Display a summary of all active trades"""
        try:
            active_trades = await self.conn.fetch("""
                SELECT t.trade_id, t.bot_id, b.name as bot_name, t.ticker, 
                       t.entry_price, t.trade_direction, t.trade_size, t.entry_time,
                       (SELECT price FROM tick_data 
                        WHERE ticker = t.ticker 
                        ORDER BY timestamp DESC LIMIT 1) as current_price
                FROM sim_bot_trades t
                JOIN sim_bots b ON t.bot_id = b.bot_id
                WHERE t.trade_status = 'open'
                ORDER BY t.entry_time DESC
            """)
            
            if not active_trades:
                logger.info(f"{Colors.YELLOW}No active trades currently{Colors.RESET}")
                return
            
            # Prepare table data
            table_data = []
            for trade in active_trades:
                current_price = float(trade['current_price'] or trade['entry_price'])
                entry_price = float(trade['entry_price'])
                
                # Calculate current P&L
                if trade['trade_direction'] == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                # Format with color indicators
                pnl_str = f"{pnl_pct:.2f}%"
                if pnl_pct > 0:
                    pnl_str = f"{Colors.GREEN}{pnl_str}{Colors.RESET}"
                elif pnl_pct < 0:
                    pnl_str = f"{Colors.RED}{pnl_str}{Colors.RESET}"
                
                direction = trade['trade_direction']
                direction_color = Colors.GREEN if direction == 'LONG' else Colors.RED
                
                # Calculate duration
                duration = datetime.now() - trade['entry_time']
                duration_str = str(duration).split('.')[0]  # Remove microseconds
                
                table_data.append([
                    trade['trade_id'],
                    f"{trade['bot_id']} ({trade['bot_name']})",
                    trade['ticker'],
                    f"{direction_color}{direction}{Colors.RESET}",
                    f"${entry_price:.2f}",
                    f"${current_price:.2f}",
                    pnl_str,
                    f"${float(trade['trade_size']):.0f}",
                    duration_str
                ])
            
            # Print table of active trades
            headers = ["ID", "Bot", "Ticker", "Direction", "Entry", "Current", "P&L", "Size", "Duration"]
            table = tabulate(table_data, headers=headers, tablefmt="pretty")
            
            logger.info(f"\n{Colors.BOLD}{Colors.BG_CYAN}{Colors.WHITE} ACTIVE TRADES SUMMARY ({len(active_trades)}) {Colors.RESET}\n{table}")
            
        except Exception as e:
            logger.error(f"Error showing active trades: {e}")
    
    async def check_rankings(self):
        """Check for changes in bot rankings"""
        try:
            rankings = await self.conn.fetch("""
                SELECT r.bot_id, b.name, b.ticker, r.rank, r.rank_score, r.is_active
                FROM bot_rankings r
                JOIN sim_bots b ON r.bot_id = b.bot_id
                ORDER BY r.rank ASC
                LIMIT 20  -- Top 20 bots
            """)
            
            if not rankings:
                logger.info(f"{Colors.YELLOW}No bot rankings found{Colors.RESET}")
                return
            
            # Detect changes in rankings
            changed_bots = []
            for bot in rankings:
                bot_id = bot['bot_id']
                if bot_id in self.rankings_cache:
                    old_rank = self.rankings_cache[bot_id]['rank']
                    new_rank = bot['rank']
                    
                    if old_rank != new_rank:
                        # Calculate rank change
                        rank_change = old_rank - new_rank  # Positive means improved rank
                        status = "improved" if rank_change > 0 else "declined"
                        
                        changed_bots.append({
                            'bot_id': bot_id,
                            'name': bot['name'],
                            'old_rank': old_rank,
                            'new_rank': new_rank,
                            'change': abs(rank_change),
                            'status': status
                        })
            
            # Print ranking changes if any
            if changed_bots:
                logger.info(f"\n{Colors.BOLD}{Colors.BG_MAGENTA}{Colors.WHITE} RANKING CHANGES DETECTED {Colors.RESET}")
                
                for bot in changed_bots:
                    status_color = Colors.GREEN if bot['status'] == "improved" else Colors.RED
                    change_symbol = "^" if bot['status'] == "improved" else "v"  # Using ASCII arrows
                    
                    logger.info(f"{Colors.BOLD}Bot #{bot['bot_id']} ({bot['name']}): "
                               f"Rank {status_color}{change_symbol}{bot['change']}{Colors.RESET} "
                               f"from #{bot['old_rank']} to #{bot['new_rank']}")
            
            # Periodically show full rankings table
            now = time.time()
            if not self.rankings_cache or now - self.last_ranking_check > 300:  # Every 5 minutes
                self.last_ranking_check = now
                
                # Prepare table data
                table_data = []
                for bot in rankings:
                    status = "Y" if bot['is_active'] else "N"  # Using ASCII characters instead of checkmarks
                    status_color = Colors.GREEN if bot['is_active'] else Colors.RED
                    
                    table_data.append([
                        bot['rank'],
                        bot['bot_id'],
                        bot['name'],
                        bot['ticker'],
                        f"{float(bot['rank_score']):.2f}",
                        f"{status_color}{status}{Colors.RESET}"
                    ])
                
                # Print rankings table
                headers = ["Rank", "Bot ID", "Name", "Ticker", "Score", "Active"]
                table = tabulate(table_data, headers=headers, tablefmt="pretty")
                
                logger.info(f"\n{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE} CURRENT BOT RANKINGS {Colors.RESET}\n{table}")
            
            # Update cache
            self.rankings_cache = {bot['bot_id']: bot for bot in rankings}
            
        except Exception as e:
            logger.error(f"Error checking rankings: {e}")
    
    async def check_metrics(self):
        """Check for significant changes in bot metrics"""
        try:
            # First check which columns exist in bot_metrics
            columns = await self.conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'bot_metrics'
            """)
            
            # Extract column names
            column_names = [col['column_name'] for col in columns]
            
            # Build a query based on available columns
            select_parts = ["m.bot_id", "b.name", "b.ticker"]
            
            # Add metrics columns if they exist
            metric_columns = ['avg_win_rate', 'win_rate', 'profit_factor', 'total_pnl', 
                             'expectancy', 'sharpe_ratio', 'max_drawdown']
            
            for col in metric_columns:
                if col in column_names:
                    select_parts.append(f"m.{col}")
                    
            # Create the query
            query = f"""
                SELECT {', '.join(select_parts)}
                FROM bot_metrics m
                JOIN sim_bots b ON m.bot_id = b.bot_id
                JOIN bot_rankings r ON m.bot_id = r.bot_id
                ORDER BY r.rank ASC
                LIMIT 10  -- Top 10 bots
            """
            
            # Get the latest metrics
            metrics = await self.conn.fetch(query)
            
            if not metrics:
                return
            
            # Log metrics columns for debugging
            logger.debug(f"Available metrics columns: {', '.join(col for col in metrics[0].keys())}")
            
            # Look for significant changes in key metrics
            significant_changes = []
            
            for bot in metrics:
                bot_id = bot['bot_id']
                
                if bot_id in self.metrics_cache:
                    old_metrics = self.metrics_cache[bot_id]
                    
                    # Check for significant changes in key metrics
                    changes = {}
                    
                    # Win rate change (more than 5 percentage points)
                    # Check different possible win rate column names
                    win_rate_col = None
                    for col in ['win_rate', 'avg_win_rate']:
                        if col in bot:
                            win_rate_col = col
                            break
                            
                    if win_rate_col and win_rate_col in old_metrics:
                        win_rate_change = float(bot[win_rate_col] or 0) - float(old_metrics[win_rate_col] or 0)
                        if abs(win_rate_change) >= 0.05:
                            changes['win_rate'] = {
                                'old': float(old_metrics[win_rate_col] or 0),
                                'new': float(bot[win_rate_col] or 0),
                                'change': win_rate_change
                            }
                    
                    # Profit factor change (more than 20%)
                    if 'profit_factor' in bot and 'profit_factor' in old_metrics:
                        if old_metrics['profit_factor'] and bot['profit_factor']:
                            pf_change = (float(bot['profit_factor']) / float(old_metrics['profit_factor'])) - 1
                            if abs(pf_change) >= 0.2:
                                changes['profit_factor'] = {
                                    'old': float(old_metrics['profit_factor']),
                                    'new': float(bot['profit_factor']),
                                    'change': pf_change
                                }
                    
                    # Total P&L change (more than $100)
                    if 'total_pnl' in bot and 'total_pnl' in old_metrics:
                        pnl_change = float(bot['total_pnl'] or 0) - float(old_metrics['total_pnl'] or 0)
                        if abs(pnl_change) >= 100:
                            changes['total_pnl'] = {
                                'old': float(old_metrics['total_pnl'] or 0),
                                'new': float(bot['total_pnl'] or 0),
                                'change': pnl_change
                            }
                    
                    if changes:
                        significant_changes.append({
                            'bot_id': bot_id,
                            'name': bot['name'],
                            'ticker': bot['ticker'],
                            'changes': changes
                        })
            
            # Report significant changes
            if significant_changes:
                logger.info(f"\n{Colors.BOLD}{Colors.BG_GREEN}{Colors.BLACK} SIGNIFICANT METRIC CHANGES {Colors.RESET}")
                
                for bot in significant_changes:
                    logger.info(f"{Colors.BOLD}Bot #{bot['bot_id']} ({bot['name']} - {bot['ticker']}){Colors.RESET}")
                    
                    for metric, data in bot['changes'].items():
                        change = data['change']
                        change_color = Colors.GREEN if change > 0 else Colors.RED
                        direction = "+" if change > 0 else ""
                        
                        if metric == 'win_rate':
                            logger.info(f"  Win Rate: {data['old']:.2%} -> {data['new']:.2%} "
                                      f"({change_color}{direction}{change:.2%}{Colors.RESET})")
                        
                        elif metric == 'profit_factor':
                            logger.info(f"  Profit Factor: {data['old']:.2f} -> {data['new']:.2f} "
                                      f"({change_color}{direction}{change:.2%}{Colors.RESET})")
                        
                        elif metric == 'total_pnl':
                            logger.info(f"  Total P&L: ${data['old']:.2f} -> ${data['new']:.2f} "
                                      f"({change_color}{direction}${abs(change):.2f}{Colors.RESET})")
            
            # Update metrics cache
            self.metrics_cache = {bot['bot_id']: bot for bot in metrics}
            
        except Exception as e:
            logger.error(f"Error checking metrics: {e}")
    
    async def show_system_health(self):
        """Display overall system health information"""
        try:
            # Get latest database stats
            db_stats = await self.conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM sim_bot_trades WHERE trade_status = 'open') as open_trades,
                    (SELECT COUNT(*) FROM sim_bot_trades WHERE trade_status = 'pending_exit') as pending_trades,
                    (SELECT COUNT(*) FROM sim_bot_trades WHERE trade_status = 'closed' AND exit_time > NOW() - INTERVAL '24 hours') as trades_24h,
                    (SELECT SUM(trade_pnl) FROM sim_bot_trades WHERE trade_status = 'closed' AND exit_time > NOW() - INTERVAL '24 hours') as pnl_24h,
                    (SELECT COUNT(*) FROM sim_bots WHERE is_active = TRUE) as active_bots,
                    (SELECT COUNT(*) FROM tick_data WHERE timestamp > NOW() - INTERVAL '5 minutes') as recent_ticks
            """)
            
            if not db_stats:
                return
            
            # Create a colorful system health report
            health_report = f"""
{Colors.BOLD}{Colors.BG_CYAN}{Colors.WHITE} SYSTEM HEALTH REPORT {Colors.RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{Colors.BOLD}Open Trades:{Colors.RESET} {Colors.YELLOW}{db_stats['open_trades']}{Colors.RESET}
{Colors.BOLD}Pending Exit Trades:{Colors.RESET} {Colors.YELLOW}{db_stats['pending_trades']}{Colors.RESET}
{Colors.BOLD}Trades (24h):{Colors.RESET} {Colors.YELLOW}{db_stats['trades_24h']}{Colors.RESET}
{Colors.BOLD}P&L (24h):{Colors.RESET} {Colors.GREEN if db_stats['pnl_24h'] and db_stats['pnl_24h'] > 0 else Colors.RED}${float(db_stats['pnl_24h'] or 0):.2f}{Colors.RESET}
{Colors.BOLD}Active Bots:{Colors.RESET} {Colors.YELLOW}{db_stats['active_bots']}{Colors.RESET}
{Colors.BOLD}Recent Ticks (5m):{Colors.RESET} {Colors.YELLOW}{db_stats['recent_ticks']}{Colors.RESET}
"""
            logger.info(health_report)
            
        except Exception as e:
            logger.error(f"Error displaying system health: {e}")
    
    async def run(self):
        """Main monitoring loop"""
        # Get the highest trade ID to start monitoring from
        try:
            self.last_trade_id = await self.conn.fetchval(
                "SELECT MAX(trade_id) FROM sim_bot_trades"
            ) or 0
            
            logger.info(f"{Colors.BOLD}{Colors.GREEN}Starting trade monitor from trade ID {self.last_trade_id}{Colors.RESET}")
            
            # Show initial system health
            await self.show_system_health()
            
            # Main monitoring loop
            while self.running:
                # Check for new trades
                await self.check_new_trades()
                
                # Periodically show active trades summary (every 30 seconds)
                current_time = time.time()
                if current_time - self.last_active_trade_check >= 30:
                    self.last_active_trade_check = current_time
                    await self.show_active_trades()
                
                # Periodically check for ranking changes (every 60 seconds)
                if current_time - self.last_ranking_check >= 60:
                    self.last_ranking_check = current_time
                    await self.check_rankings()
                
                # Periodically check for metric changes (every 5 minutes)
                if current_time - self.last_metrics_check >= 300:
                    self.last_metrics_check = current_time
                    await self.check_metrics()
                    await self.show_system_health()
                
                # Sleep to prevent excessive database queries
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main monitoring loop: {e}")
        finally:
            logger.info("Monitoring stopped")

def setup_windows_console():
    """Try to configure Windows console for proper color support"""
    if sys.platform == 'win32':
        try:
            # Try to enable ANSI escape sequences in Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            
            # Set console to use UTF-8
            import subprocess
            subprocess.run(["chcp", "65001"], shell=True, check=False)
            
            # Also set stdout to use UTF-8 encoding
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Could not configure Windows console for color support: {e}")
            print("Some formatting may not display correctly.")

async def main():
    """Main function"""
    # Try to configure Windows console for color support
    setup_windows_console()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Print banner
    print(f"""
{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE} KNOW-DEFEAT TRADE MONITOR {Colors.RESET}

{Colors.CYAN}Monitoring trades, rankings, and metrics in real-time
Press Ctrl+C to exit{Colors.RESET}
""")
    
    # Create and run monitor
    monitor = TradeMonitor()
    
    try:
        # Connect to database
        if await monitor.connect():
            # Run monitoring loop
            await monitor.run()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    finally:
        # Close database connection
        await monitor.disconnect()

if __name__ == "__main__":
    asyncio.run(main())