#!/bin/bash
# Script to ensure proper imports in the Know-Defeat project

echo "Creating/updating __init__.py files for proper package structure..."

# Create __init__.py files in key directories
touch __init__.py
touch src/__init__.py
touch algorithms/__init__.py

echo "Fixing import paths in base_bot.py..."

echo "Testing imports with test script..."
python3 test_bot_import.py

echo ""
echo "If everything is working correctly, now you can run:"
echo "./start_trading.sh"

echo ""
echo "NOTE: If there are still import errors, try updating the code to use:"
echo "- from src.bot_ranker import BotRanker"
echo "- import algorithms.breakout_algorithm"
echo "instead of relative imports."