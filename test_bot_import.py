#!/usr/bin/env python3
"""
Direct test script for importing and creating a bot instance.
Run this directly from the project root.
"""

import os
import sys
import importlib
import asyncio
import yaml

# Add both the project root and src directory to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

print(f"Testing from project root: {project_root}")
print(f"Python path: {sys.path}")

async def test_bot_imports():
    print("\n1. Testing import of bot_ranker.py:")
    try:
        # Try different import combinations
        try:
            import bot_ranker
            print("✓ Successfully imported bot_ranker directly")
        except ImportError:
            try:
                import src.bot_ranker
                print("✓ Successfully imported src.bot_ranker")
            except ImportError:
                from src import bot_ranker
                print("✓ Successfully imported bot_ranker from src")
    except ImportError as e:
        print(f"✗ Failed to import bot_ranker: {e}")
    
    print("\n2. Testing import of base_bot.py:")
    try:
        try:
            import base_bot
            print("✓ Successfully imported base_bot directly")
            BotFactory = base_bot.BotFactory
        except ImportError:
            try:
                import src.base_bot
                print("✓ Successfully imported src.base_bot")
                BotFactory = src.base_bot.BotFactory
            except ImportError:
                from src import base_bot
                print("✓ Successfully imported base_bot from src")
                BotFactory = base_bot.BotFactory
    except ImportError as e:
        print(f"✗ Failed to import base_bot: {e}")
        return
    
    print("\n3. Testing import of algorithm modules:")
    for algo in ["momentum_algorithm", "breakout_algorithm"]:
        try:
            module = importlib.import_module(f"algorithms.{algo}")
            print(f"✓ Successfully imported algorithms.{algo}")
            print(f"  Module file: {module.__file__}")
            for name in dir(module):
                if name.endswith('Algorithm'):
                    print(f"  Found class: {name}")
        except ImportError as e:
            print(f"✗ Failed to import algorithms.{algo}: {e}")
    
    print("\n4. Testing loading a specific bot YAML file:")
    test_yaml = os.path.join(project_root, 'src', 'bots', '17_TSLA_breakout.yaml')
    if os.path.exists(test_yaml):
        try:
            with open(test_yaml, 'r') as f:
                config = yaml.safe_load(f)
                print(f"✓ Successfully loaded {test_yaml}")
                print(f"  Bot name: {config.get('name')}")
                print(f"  Algorithm module: {config.get('algorithm_module')}")
                
                # Try to import this specific algorithm
                module_path = config.get('algorithm_module')
                try:
                    module = importlib.import_module(module_path)
                    print(f"✓ Successfully imported {module_path}")
                    print(f"  Module file: {module.__file__}")
                except ImportError as e:
                    print(f"✗ Failed to import {module_path}: {e}")
        except Exception as e:
            print(f"✗ Failed to load YAML: {e}")
    else:
        print(f"✗ Test YAML file not found: {test_yaml}")

if __name__ == "__main__":
    asyncio.run(test_bot_imports())