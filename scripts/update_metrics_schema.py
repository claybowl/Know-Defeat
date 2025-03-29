#!/usr/bin/env python3
"""
Script to update the bot_metrics table schema by adding the rank_score column
"""

import psycopg2
import os
import argparse

def update_metrics_schema(host='localhost', port=5432, dbname='tick_data', user='clayb', password='musicman'):
    """
    Executes the SQL to add the rank_score column to the bot_metrics table
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the SQL file
    sql_file_path = os.path.join(script_dir, 'add_rank_score_column.sql')
    
    # Check if SQL file exists
    if not os.path.exists(sql_file_path):
        print(f"ERROR: SQL file not found at {sql_file_path}")
        return False
    
    # Read SQL content
    with open(sql_file_path, 'r') as file:
        sql_content = file.read()
    
    # Connect to PostgreSQL
    print(f"Connecting to PostgreSQL database: {dbname} on {host}:{port}")
    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        # Set autocommit mode
        conn.autocommit = True
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Execute the SQL
        print("Executing SQL to update the bot_metrics table schema...")
        cursor.execute(sql_content)
        
        # Check for any output messages
        result = cursor.fetchall() if cursor.description else None
        if result:
            for row in result:
                print(row)
        
        print("Database schema update completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to update database schema: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Update the bot_metrics table schema.')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--dbname', default='tick_data', help='Database name')
    parser.add_argument('--user', default='clayb', help='Database user')
    parser.add_argument('--password', default='musicman', help='Database password')
    
    args = parser.parse_args()
    
    update_metrics_schema(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password
    )

if __name__ == "__main__":
    main()