import asyncio
import asyncpg
import sys

async def check_database_schema():
    try:
        print("Connecting to database...")
        # Connect to the database
        conn = await asyncpg.connect(
            user='clayb',
            password='musicman',
            database='tick_data',
            host='localhost'
        )
        
        # First check if the table exists
        exists = await conn.fetchval("""
        SELECT EXISTS(
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = 'bot_metrics'
        );
        """)
        
        if not exists:
            print("The table 'bot_metrics' does not exist in the database.")
            
            # Let's check what tables do exist
            print("\nExisting tables in the database:")
            tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """)
            
            for table in tables:
                print(f"- {table['table_name']}")
                
            await conn.close()
            return
        
        # Get table schema information
        columns = await conn.fetch("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = 'bot_metrics'
        ORDER BY ordinal_position;
        """)
        
        if columns:
            print("\nBot Metrics Table Schema:")
            for column in columns:
                if column['character_maximum_length']:
                    print(f"{column['column_name']}: {column['data_type']}({column['character_maximum_length']})")
                else:
                    print(f"{column['column_name']}: {column['data_type']}")
        else:
            print("Table 'bot_metrics' exists but has no columns.")
        
        # Check for constraints
        constraints = await conn.fetch("""
        SELECT 
            tc.constraint_name, 
            tc.constraint_type,
            kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = 'bot_metrics'
        ORDER BY tc.constraint_name, kcu.column_name;
        """)
        
        if constraints:
            print("\nConstraints:")
            for constraint in constraints:
                print(f"{constraint['constraint_name']} ({constraint['constraint_type']}): {constraint['column_name']}")
        
        # Check for bot_metrics records count
        count = await conn.fetchval("SELECT COUNT(*) FROM bot_metrics")
        print(f"\nTotal records in bot_metrics: {count}")
        
        # Check for metrics in the variable_weights table
        exists = await conn.fetchval("""
        SELECT EXISTS(
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_name = 'variable_weights'
        );
        """)
        
        if exists:
            weights = await conn.fetch("SELECT variable_name, weight FROM variable_weights ORDER BY variable_name")
            print("\nVariable Weights:")
            for w in weights:
                print(f"{w['variable_name']}: {w['weight']}")
        
        await conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# Run the async function
asyncio.run(check_database_schema()) 