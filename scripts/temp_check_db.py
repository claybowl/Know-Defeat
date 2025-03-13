import asyncio
import asyncpg

async def check_database():
    # Connect to the database
    conn = await asyncpg.connect(
        host='localhost',
        database='tick_data',
        user='clayb',
        password='musicman'
    )
    
    # Check what tables exist
    print("Tables in the database:")
    tables = await conn.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    for table in tables:
        print(f"- {table['table_name']}")
        
        # Get row count for each table
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
        print(f"  Count: {count}")
        
        # Get sample data (first row) for small tables
        if count > 0 and count < 1000:
            try:
                sample = await conn.fetch(f"SELECT * FROM {table['table_name']} LIMIT 1")
                print(f"  Sample: {sample}")
            except:
                print(f"  Could not fetch sample")
    
    # Check bot_metrics table specifically
    if any(table['table_name'] == 'bot_metrics' for table in tables):
        print("\nBot Metrics Structure:")
        columns = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'bot_metrics'
            ORDER BY ordinal_position;
        """)
        
        for column in columns:
            print(f"- {column['column_name']}: {column['data_type']}")
    
    # Check bot_rankings table
    if any(table['table_name'] == 'bot_rankings' for table in tables):
        print("\nBot Rankings Structure:")
        columns = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'bot_rankings'
            ORDER BY ordinal_position;
        """)
        
        for column in columns:
            print(f"- {column['column_name']}: {column['data_type']}")
            
        # Check if there are any bot rankings
        rankings = await conn.fetch("""
            SELECT bot_id, ticker, rank_score, is_active 
            FROM bot_rankings
            ORDER BY rank_score DESC
            LIMIT 10
        """)
        
        print("\nTop 10 Bot Rankings:")
        for rank in rankings:
            print(f"Bot {rank['bot_id']} ({rank['ticker']}): {rank['rank_score']}, Active: {rank['is_active']}")
    
    # Close the connection
    await conn.close()

# Run the async function
asyncio.run(check_database())