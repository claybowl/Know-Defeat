#!/bin/bash

# Delete trades for specified bots
echo "Deleting trades for bots 2, 3, and 102..."

# Run the SQL script
psql -U clayb -d tick_data -f scripts/delete_bot_trades.sql

echo "Operation completed."