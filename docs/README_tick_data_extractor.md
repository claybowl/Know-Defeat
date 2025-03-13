# Tick Data Extractor

This script extracts ticker data from the `ib_controller_simple.log` file and saves it to a CSV file for further analysis and charting.

## Usage

Run the script from the command line:

```bash
# Basic usage - will try to automatically find the log file
python scripts/extract_tick_data.py

# Specify log file location
python scripts/extract_tick_data.py --log /path/to/ib_controller_simple.log

# Specify both log file and output CSV file
python scripts/extract_tick_data.py --log /path/to/ib_controller_simple.log --output /path/to/output.csv
```

### Command-line Arguments

- `-l, --log`: Path to the log file (optional, will try to auto-detect if not specified)
- `-o, --output`: Path to save the CSV file (optional, defaults to `tick_data_YYYYMMDD_HHMMSS.csv` in the project root)

## What it does

1. The script tries to find the `ib_controller_simple.log` file in common locations or uses the path provided
2. It looks for log entries matching the pattern: `YYYY-MM-DD HH:MM:SS,SSS - INFO - Tick Price - {ticker}: LAST = ${last_price}`
3. Extracts the timestamp, ticker symbol, and last price from each matching entry
4. Saves the extracted data to a CSV file

## Auto-detection of Log Files

The script will attempt to find the log file in the following locations:
- Project root directory
- `logs/` directory
- `logs/app_logs/` directory
- `logs/trade_logs/` directory
- `src/` directory

## Output Format

The CSV file contains three columns:
- `Timestamp`: The timestamp from the log (format: YYYY-MM-DD HH:MM:SS,SSS)
- `Ticker`: The ticker symbol
- `Last Price`: The last price of the ticker

## Troubleshooting

- If the script fails to find the log file automatically, use the `--log` argument to specify the exact path
- If no data is extracted, check that the log file contains entries matching the expected format
- Make sure you have proper read/write permissions for the log file and the output directory

## Example Output

```
Timestamp,Ticker,Last Price
2025-03-12 17:32:34,888,AAPL,175.23
2025-03-12 17:32:35,123,MSFT,410.34
2025-03-12 17:32:36,456,TSLA,180.56
``` 