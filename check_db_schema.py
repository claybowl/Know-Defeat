import psycopg2
import sys
import os

try:
    # Try to run the conda activation command to make sure we're in the right environment
    print("Running script to check bot_metrics table schema...")
    
    # Connect to the database
    conn = psycopg2.connect(
        dbname="tick_data",
        user="clayb",
        password="musicman",
        host="localhost"
    )
    
    # Create a cursor
    cur = conn.cursor()
    
    # First check if the table exists
    cur.execute("""
    SELECT EXISTS(
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_name = 'bot_metrics'
    );
    """)
    
    table_exists = cur.fetchone()[0]
    
    if not table_exists:
        print("The table 'bot_metrics' does not exist in the database.")
        
        # Let's check what tables do exist
        print("\nExisting tables in the database:")
        cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """)
        
        tables = cur.fetchall()
        for table in tables:
            print(f"- {table[0]}")
        
        sys.exit(0)
    
    # Get table schema information
    cur.execute("""
    SELECT column_name, data_type, character_maximum_length
    FROM information_schema.columns
    WHERE table_name = 'bot_metrics'
    ORDER BY ordinal_position;
    """)
    
    # Fetch results
    columns = cur.fetchall()
    
    if columns:
        print("Bot Metrics Table Schema:")
        for column in columns:
            if column[2]:  # If it has a character maximum length
                print(f"{column[0]}: {column[1]}({column[2]})")
            else:
                print(f"{column[0]}: {column[1]}")
    else:
        print("Table 'bot_metrics' exists but has no columns.")
    
    # Check for constraints/primary keys/unique constraints
    cur.execute("""
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
    
    constraints = cur.fetchall()
    if constraints:
        print("\nConstraints:")
        for constraint in constraints:
            print(f"{constraint[0]} ({constraint[1]}): {constraint[2]}")
    
    # Close cursor and connection
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 