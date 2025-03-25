"""
Simple system check script for the Know-Defeat trading system.
This script prints instructions and commands to check your system.
"""

print("=== Know-Defeat Trading System Check ===")
print("This script helps you check if your trading system is set up correctly.")
print("Follow these steps to verify your system:")

print("\n1. Check if PostgreSQL is running:")
print("   - Windows: Open Services and look for 'postgresql'")
print("   - Linux/Mac: Run 'ps aux | grep postgres'")

print("\n2. Connect to your database:")
print("   - Run: psql -U clayb -d tick_data")

print("\n3. Run these queries to check your database:")
print("""
   - Check bots:
     SELECT bot_id, ticker, algorithm_type FROM sim_bots ORDER BY bot_id LIMIT 10;
   
   - Check trades:
     SELECT trade_id, bot_id, ticker, trade_direction, trade_pnl, entry_time, exit_time 
     FROM sim_bot_trades ORDER BY entry_time DESC LIMIT 5;
   
   - Check metrics:
     SELECT bot_id, timestamp, total_trades, avg_win_rate, total_pnl 
     FROM bot_metrics ORDER BY timestamp DESC LIMIT 5;
""")

print("\n4. To create a test trade and update metrics, run:")
print("   - First, activate your Anaconda environment:")
print("     conda activate Autogen")
print("   - Then run the test script:")
print("     python tests/test_trade_creation.py")

print("\n5. To check the metrics calculation system, run:")
print("   - conda activate Autogen")
print("   - python tests/test_metrics_system.py")

print("\n6. To test the full trading pipeline, run:")
print("   - conda activate Autogen")
print("   - python tests/test_trading_pipeline.py")

print("\nRequired Python packages:")
print("   - asyncpg: For async database connections")
print("   - psycopg2: For synchronous database connections")
print("   - pandas: For data manipulation")
print("   - tabulate: For nice table formatting")

print("\nInstall missing packages with:")
print("   conda install -c conda-forge asyncpg psycopg2 pandas tabulate")
print("   or")
print("   pip install asyncpg psycopg2-binary pandas tabulate")

print("\nIf you're having issues with database connections, make sure:")
print("1. PostgreSQL is running")
print("2. Your database credentials are correct")
print("3. The tables exist in your database")

print("\nTo check table structures, run these SQL commands:")
print("""
   - Check sim_bots table:
     \d sim_bots
   
   - Check sim_bot_trades table:
     \d sim_bot_trades
   
   - Check bot_metrics table:
     \d bot_metrics
""")

print("\nGood luck with your testing!")