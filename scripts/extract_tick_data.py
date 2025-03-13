#!/usr/bin/env python3
import re
import csv
import os
import argparse
from datetime import datetime

def extract_tick_data(log_file_path, output_csv_path):
    """
    Extract tick data from the log file and save it to a CSV file.
    
    Args:
        log_file_path: Path to the log file
        output_csv_path: Path to save the CSV file
    """
    # Pattern to match the log entries
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Tick Price - (.+?): LAST = \$(.+?)$'
    
    # Lists to store the extracted data
    timestamps = []
    tickers = []
    prices = []
    
    # Read the log file and extract data
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                match = re.match(pattern, line.strip())
                if match:
                    timestamp, ticker, price = match.groups()
                    timestamps.append(timestamp)
                    tickers.append(ticker)
                    prices.append(price)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return False
    except Exception as e:
        print(f"Error reading log file: {e}")
        return False
    
    # Check if any data was extracted
    if not timestamps:
        print("No matching data found in the log file.")
        return False
    
    # Write the extracted data to a CSV file
    try:
        with open(output_csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(['Timestamp', 'Ticker', 'Last Price'])
            # Write the data
            for i in range(len(timestamps)):
                writer.writerow([timestamps[i], tickers[i], prices[i]])
        
        print(f"Successfully extracted {len(timestamps)} tick data entries to {output_csv_path}")
        return True
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        return False

def find_log_file(base_dir, filename="ib_controller_simple.log"):
    """
    Attempts to find the log file by searching common locations
    
    Args:
        base_dir: Base directory to start the search
        filename: Name of the log file to search for
        
    Returns:
        Path to the log file if found, None otherwise
    """
    # Common subdirectories to check
    common_dirs = [
        "",  # Root directory
        "logs",
        "logs/app_logs",
        "logs/trade_logs",
        "src"
    ]
    
    # Check all common directories
    for dir_path in common_dirs:
        full_path = os.path.join(base_dir, dir_path, filename)
        if os.path.isfile(full_path):
            print(f"Found log file at: {full_path}")
            return full_path
    
    return None

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract tick data from log file and save to CSV")
    parser.add_argument('-l', '--log', help="Path to the log file (default: tries to find ib_controller_simple.log)", default=None)
    parser.add_argument('-o', '--output', help="Path to save the CSV file (default: tick_data_TIMESTAMP.csv in the current directory)", default=None)
    args = parser.parse_args()
    
    # Set base directory (project root)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine log file path
    if args.log:
        # User specified log file path
        log_file_path = args.log
    else:
        # Try to find the log file
        log_file_path = find_log_file(current_dir)
        if not log_file_path:
            print("Error: Could not find the log file. Please specify the path using the --log argument.")
            print("Example: python extract_tick_data.py --log /path/to/ib_controller_simple.log")
            exit(1)
    
    # Determine output CSV path
    if args.output:
        output_csv_path = args.output
    else:
        # Generate a filename with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(current_dir, f"tick_data_{timestamp}.csv")
    
    # Extract the data
    extract_tick_data(log_file_path, output_csv_path) 