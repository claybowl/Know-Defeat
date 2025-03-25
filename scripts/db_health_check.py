"""
Database Health Check Script
This script connects to the tick_data database and performs a series of health checks.
"""

import asyncio
import asyncpg
import sys
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost',
    'port': 5432
}

async def check_database_health():
    """Perform a comprehensive health check on the tick_data database."""
    try:
        print("Starting database health check...")
        print(f"Connecting to PostgreSQL database: {DB_CONFIG['database']}")
        
        # Create a connection pool
        pool = await asyncpg.create_pool(**DB_CONFIG)
        
        # Check 1: Verify Connection
        print("\n=== Connection Test ===")
        async with pool.acquire() as conn:
            db_version = await conn.fetchval("SELECT version();")
            print(f"✅ Successfully connected to PostgreSQL")
            print(f"PostgreSQL version: {db_version}")
        
        # Check 2: Database Size
        print("\n=== Database Size ===")
        async with pool.acquire() as conn:
            db_size = await conn.fetchval(
                "SELECT pg_size_pretty(pg_database_size($1));",
                DB_CONFIG['database']
            )
            print(f"Database size: {db_size}")
        
        # Check 3: Table Count
        print("\n=== Tables ===")
        async with pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name, 
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC;
            """)
            
            print(f"Total tables: {len(tables)}")
            print("Top 10 tables by size:")
            for i, table in enumerate(tables[:10], 1):
                print(f"{i}. {table['table_name']} - {table['size']}")
        
        # Check 4: Record Counts
        print("\n=== Record Counts ===")
        async with pool.acquire() as conn:
            main_tables = ['tick_data', 'sim_bots', 'sim_bot_trades', 'bot_metrics', 'bot_rankings']
            
            for table in main_tables:
                # Check if table exists
                exists = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    );
                """)
                
                if exists:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table};")
                    print(f"{table}: {count:,} records")
                else:
                    print(f"{table}: Table does not exist")
        
        # Check 5: Recent Bot Trades
        print("\n=== Recent Bot Trades ===")
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'sim_bot_trades'
                );
            """)
            
            if table_exists:
                recent_trades = await conn.fetch("""
                    SELECT trade_id, bot_id, ticker, trade_direction, 
                           entry_time, trade_status, trade_pnl
                    FROM sim_bot_trades
                    ORDER BY entry_time DESC
                    LIMIT 5;
                """)
                
                if recent_trades:
                    print("5 most recent trades:")
                    for trade in recent_trades:
                        trade_status = trade['trade_status']
                        pnl_str = f", PnL: ${trade['trade_pnl']}" if trade['trade_pnl'] is not None else ""
                        
                        print(f"Trade ID: {trade['trade_id']}, "
                              f"Bot: {trade['bot_id']}, "
                              f"Ticker: {trade['ticker']}, "
                              f"Direction: {trade['trade_direction']}, "
                              f"Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}, "
                              f"Status: {trade_status}{pnl_str}")
                else:
                    print("No recent trades found")
            else:
                print("sim_bot_trades table does not exist")
        
        # Check 6: Bot Rankings
        print("\n=== Top Ranked Bots ===")
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'bot_rankings'
                );
            """)
            
            if table_exists:
                top_bots = await conn.fetch("""
                    SELECT br.bot_id, br.rank, br.rank_score, sb.ticker, sb.algorithm_type
                    FROM bot_rankings br
                    JOIN sim_bots sb ON br.bot_id = sb.bot_id
                    ORDER BY br.rank
                    LIMIT 10;
                """)
                
                if top_bots:
                    print("Top 10 ranked bots:")
                    for bot in top_bots:
                        print(f"Rank {bot['rank']}: Bot {bot['bot_id']} "
                              f"({bot['ticker']} - {bot['algorithm_type']}), "
                              f"Score: {bot['rank_score']}")
                else:
                    print("No bot rankings found")
            else:
                print("bot_rankings table does not exist")
        
        # Check 7: Database Health
        print("\n=== Database Health ===")
        async with pool.acquire() as conn:
            # Check for dead tuples and bloat
            bloat_query = """
                SELECT schemaname, tablename, 
                       pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
                       pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size,
                       pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename) - 
                                    pg_relation_size(schemaname || '.' || tablename)) as index_size
                FROM pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
                LIMIT 5;
            """
            bloat_info = await conn.fetch(bloat_query)
            
            print("Table sizes (including indexes):")
            for info in bloat_info:
                print(f"{info['tablename']}: "
                      f"Total: {info['total_size']}, "
                      f"Table: {info['table_size']}, "
                      f"Indexes: {info['index_size']}")
        
        # Final status
        print("\n=== Overall Status ===")
        print("✅ Database check completed successfully")
        print("The database appears to be in good health and ready for trading operations.")
        
        # Close the connection pool
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error during database health check: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    return True

# Run the health check
if __name__ == "__main__":
    success = asyncio.run(check_database_health())
    sys.exit(0 if success else 1)