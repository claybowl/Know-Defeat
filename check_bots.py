import os
import json
import re
from datetime import datetime

def check_bots_manually():
    """
    Analyze bots 1, 5, 7, and 103 by examining files in the project.
    This is a fallback approach since direct database access is not working.
    """
    print("Analyzing bots 1, 5, 7, and 103 through project files...")
    
    # Bots to analyze
    bot_ids = [1, 5, 7, 103]
    results = {bot_id: {} for bot_id in bot_ids}
    
    # 1. Let's check any log files with metrics data
    log_dir = os.path.join(os.getcwd(), "logs")
    if os.path.exists(log_dir):
        print("\n=== Checking Log Files ===")
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        metrics_data = {}
        
        for log_file in log_files:
            print(f"Examining {log_file}...")
            try:
                with open(os.path.join(log_dir, log_file), 'r') as f:
                    content = f.read()
                    
                    # Look for metrics related to our bots
                    for bot_id in bot_ids:
                        # Try to find metrics for this bot
                        bot_pattern = rf"Bot\s+{bot_id}\b.*?\b(win_rate|avg_win_rate|profit_factor|total_pnl)\b.*?(\d+\.?\d*)"
                        matches = re.findall(bot_pattern, content, re.IGNORECASE)
                        
                        if matches:
                            if bot_id not in metrics_data:
                                metrics_data[bot_id] = {}
                            
                            for metric, value in matches:
                                metrics_data[bot_id][metric.lower()] = value
                            
                            # Also look for warnings
                            warning_pattern = rf"Bot\s+{bot_id}\b.*?(warning|error|issue|discrepancy)"
                            warnings = re.findall(warning_pattern, content, re.IGNORECASE)
                            
                            if warnings:
                                if 'warnings' not in metrics_data[bot_id]:
                                    metrics_data[bot_id]['warnings'] = []
                                metrics_data[bot_id]['warnings'].extend(warnings)
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        # Report findings from logs
        for bot_id in bot_ids:
            if bot_id in metrics_data:
                print(f"\nBot {bot_id} metrics from logs:")
                for metric, value in metrics_data[bot_id].items():
                    if metric != 'warnings':
                        print(f"  {metric}: {value}")
                
                if 'warnings' in metrics_data[bot_id]:
                    print("  Warnings found:")
                    for warning in metrics_data[bot_id]['warnings']:
                        print(f"  - {warning}")
            else:
                print(f"\nBot {bot_id}: No metrics found in logs")
    else:
        print("Logs directory not found")
    
    # 2. Look for any configuration files that might have info on these bots
    config_files = []
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith(('.json', '.yaml', '.yml')) and 'bot' in file.lower():
                config_files.append(os.path.join(root, file))
    
    if config_files:
        print("\n=== Checking Configuration Files ===")
        for config_file in config_files:
            print(f"Examining {os.path.basename(config_file)}...")
            try:
                with open(config_file, 'r') as f:
                    # Try to parse as JSON first
                    try:
                        data = json.load(f)
                        
                        # Check if this contains bot configurations
                        if isinstance(data, list):
                            # Maybe a list of bot configs
                            for item in data:
                                if isinstance(item, dict) and 'bot_id' in item:
                                    bot_id = item['bot_id']
                                    if bot_id in bot_ids:
                                        print(f"Found configuration for Bot {bot_id}:")
                                        for key, value in item.items():
                                            print(f"  {key}: {value}")
                                        results[bot_id]['config'] = item
                        elif isinstance(data, dict):
                            # Check if it's a single bot config
                            if 'bot_id' in data and data['bot_id'] in bot_ids:
                                bot_id = data['bot_id']
                                print(f"Found configuration for Bot {bot_id}:")
                                for key, value in data.items():
                                    print(f"  {key}: {value}")
                                results[bot_id]['config'] = data
                            # Check if it's a mapping by bot_id
                            else:
                                for bot_id_str, bot_data in data.items():
                                    try:
                                        bot_id = int(bot_id_str)
                                        if bot_id in bot_ids and isinstance(bot_data, dict):
                                            print(f"Found configuration for Bot {bot_id}:")
                                            for key, value in bot_data.items():
                                                print(f"  {key}: {value}")
                                            results[bot_id]['config'] = bot_data
                                    except ValueError:
                                        # Not a numeric key
                                        continue
                    except json.JSONDecodeError:
                        # Not a valid JSON file, skip
                        pass
            except Exception as e:
                print(f"Error reading {config_file}: {e}")
    else:
        print("No bot configuration files found")
    
    # 3. Check Python files that might define these bots
    print("\n=== Checking Python Files for Bot Definitions ===")
    source_files = []
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.py'):
                source_files.append(os.path.join(root, file))
    
    bot_definitions = {}
    for source_file in source_files:
        try:
            with open(source_file, 'r') as f:
                content = f.read()
                
                # Look for bot definitions
                for bot_id in bot_ids:
                    # Pattern to match bot configuration definitions
                    bot_pattern = rf"bot_id\s*[=:]\s*{bot_id}\b"
                    if re.search(bot_pattern, content):
                        print(f"Found bot {bot_id} definition in {os.path.basename(source_file)}")
                        
                        # Try to extract parameters
                        param_pattern = r"(\w+)\s*[=:]\s*[\'\"]?([^\'\",]+)[\'\"]?"
                        
                        # Find the bot definition block (approximation)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if re.search(bot_pattern, line):
                                # Extract a block of ~15 lines
                                block = '\n'.join(lines[max(0, i-2):min(len(lines), i+15)])
                                
                                # Extract parameters
                                params = re.findall(param_pattern, block)
                                if params:
                                    if bot_id not in bot_definitions:
                                        bot_definitions[bot_id] = {}
                                    
                                    print(f"  Parameters found:")
                                    for param, value in params:
                                        bot_definitions[bot_id][param] = value
                                        print(f"    {param}: {value}")
                        
                        # Try to detect any unusual parameters or values
                        if bot_id in bot_definitions:
                            if 'win_rate' in bot_definitions[bot_id]:
                                win_rate = float(bot_definitions[bot_id]['win_rate'])
                                if win_rate > 0.95:
                                    print(f"  WARNING: Unusually high win rate: {win_rate}")
                            
                            if 'profit_factor' in bot_definitions[bot_id]:
                                profit_factor = float(bot_definitions[bot_id]['profit_factor'])
                                if profit_factor > 50:
                                    print(f"  WARNING: Extreme profit factor: {profit_factor}")
        except Exception as e:
            # Skip files that can't be read
            continue
    
    print("\n=== Summary of Findings ===")
    for bot_id in bot_ids:
        print(f"\nBot {bot_id}:")
        
        # Collect all information from various sources
        info = {}
        info.update(results.get(bot_id, {}))
        if bot_id in metrics_data:
            info['metrics'] = metrics_data[bot_id]
        if bot_id in bot_definitions:
            info['code_definition'] = bot_definitions[bot_id]
        
        if not info:
            print("  No information found for this bot")
            continue
        
        # Check for suspicious metrics
        warnings = []
        
        # Check win rate
        win_rate = None
        for source in ['metrics', 'config', 'code_definition']:
            if source in info:
                if 'win_rate' in info[source]:
                    win_rate = float(info[source]['win_rate'])
                    print(f"  Win Rate: {win_rate}% (from {source})")
                    break
                elif 'avg_win_rate' in info[source]:
                    win_rate = float(info[source]['avg_win_rate'])
                    print(f"  Win Rate: {win_rate}% (from {source})")
                    break
        
        if win_rate and win_rate > 95:
            warnings.append(f"Suspiciously high win rate: {win_rate}%")
        
        # Check profit factor
        profit_factor = None
        for source in ['metrics', 'config', 'code_definition']:
            if source in info:
                if 'profit_factor' in info[source]:
                    profit_factor = float(info[source]['profit_factor'])
                    print(f"  Profit Factor: {profit_factor} (from {source})")
                    break
        
        if profit_factor and profit_factor > 50:
            warnings.append(f"Extreme profit factor: {profit_factor}")
        
        # Report all warnings
        if warnings:
            print("  ISSUES DETECTED:")
            for warning in warnings:
                print(f"  - {warning}")
        
        # Report overall assessment
        if warnings:
            print("  RECOMMENDATION: This bot's metrics look suspicious and should be investigated")
        else:
            print("  No obvious issues detected with this bot based on available information")
    
    print("\nAnalysis complete")

# Run the analysis
if __name__ == "__main__":
    check_bots_manually() 