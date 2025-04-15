"""
Test script for the improved metrics calculator.

This script tests the new EnhancedMetricsCalculator against known test data
and compares the results with both the old calculator and expected values.
"""

import asyncio
import asyncpg
import logging
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import both the original and improved calculators
from src.metrics_calculator import MetricsCalculator
from src.metrics_calculator_improvements import EnhancedMetricsCalculator

# Reuse the test data setup from metrics validation
from test_metrics_validation import setup_test_data, calculate_expected_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'user': 'clayb',
    'password': 'musicman',
    'database': 'tick_data',
    'host': 'localhost'
}

async def run_original_calculator(pool, test_bot_id, test_ticker):
    """Run the original metrics calculator."""
    logger.info(f"Running original metrics calculator for bot {test_bot_id}")
    
    metrics_calculator = MetricsCalculator(pool)
    original_metrics = {}
    
    # Gather metrics from original calculator
    original_metrics["one_hour_performance"] = await metrics_calculator.calculate_one_hour_performance(test_bot_id, test_ticker)
    original_metrics["two_hour_performance"] = await metrics_calculator.calculate_two_hour_performance(test_bot_id, test_bot_id)
    original_metrics["avg_win_rate"] = await metrics_calculator.calculate_avg_win_rate(test_bot_id, test_ticker)
    original_metrics["total_pnl"] = await metrics_calculator.calculate_total_pnl(test_bot_id, test_bot_id)
    original_metrics["avg_profit_per_trade"] = await metrics_calculator.calculate_avg_profit_per_trade(test_bot_id, test_bot_id)
    original_metrics["profit_factor"] = await metrics_calculator.calculate_profit_factor(test_bot_id, test_bot_id)
    
    # Get drawdown metrics
    drawdown_info = await metrics_calculator.calculate_drawdowns(test_bot_id, test_bot_id)
    original_metrics["avg_drawdown"] = drawdown_info["avg_drawdown"]
    original_metrics["max_drawdown"] = drawdown_info["max_drawdown"]
    
    # Get other metrics
    original_metrics["profit_per_second"] = await metrics_calculator.calculate_profit_per_second(test_bot_id, test_bot_id)
    original_metrics["sharpe_ratio"] = await metrics_calculator.calculate_sharpe_ratio(test_bot_id, test_bot_id)
    
    # Format all numeric values to be consistent
    for key, value in original_metrics.items():
        if isinstance(value, (int, float)):
            original_metrics[key] = round(float(value), 4)
    
    return original_metrics

async def run_enhanced_calculator(pool, test_bot_id, test_ticker):
    """Run the enhanced metrics calculator."""
    logger.info(f"Running enhanced metrics calculator for bot {test_bot_id}")
    
    enhanced_calculator = EnhancedMetricsCalculator(pool)
    
    # Get all metrics at once
    enhanced_metrics = await enhanced_calculator.calculate_all_metrics(test_bot_id, test_ticker)
    
    # Format all numeric values to be consistent
    for key, value in enhanced_metrics.items():
        if isinstance(value, (int, float)):
            enhanced_metrics[key] = round(float(value), 4)
    
    return enhanced_metrics

def compare_calculators(expected, original, enhanced):
    """Compare the results of both calculators against expected values."""
    logger.info("Comparing calculator results")
    
    comparison_data = []
    
    # Combine all metric keys
    all_metrics = set(expected.keys()) | set(original.keys()) | set(enhanced.keys())
    
    for metric in sorted(all_metrics):
        expected_value = expected.get(metric, "N/A")
        original_value = original.get(metric, "N/A")
        enhanced_value = enhanced.get(metric, "N/A")
        
        # Determine improvement status
        if metric in expected and metric in enhanced and metric in original:
            if isinstance(expected_value, (int, float)) and isinstance(enhanced_value, (int, float)) and isinstance(original_value, (int, float)):
                exp_val = float(expected_value)
                orig_val = float(original_value)
                enh_val = float(enhanced_value)
                
                # Calculate relative errors
                orig_diff = abs(exp_val - orig_val)
                enh_diff = abs(exp_val - enh_val)
                
                orig_rel_error = (orig_diff / max(abs(exp_val), 0.0001)) * 100 if exp_val != 0 else (0 if orig_val == 0 else 100)
                enh_rel_error = (enh_diff / max(abs(exp_val), 0.0001)) * 100 if exp_val != 0 else (0 if enh_val == 0 else 100)
                
                # Determine improvement status
                if enh_rel_error < orig_rel_error:
                    status = "✅ Improved"
                elif enh_rel_error == orig_rel_error:
                    status = "🟰 Same"
                else:
                    status = "❌ Regression"
                
                improvement = f"{orig_rel_error:.1f}% → {enh_rel_error:.1f}%"
            else:
                status = "❓ Unknown"
                improvement = "N/A"
        elif metric in enhanced and metric not in original:
            status = "➕ New"
            improvement = "N/A"
        else:
            status = "❓ Unknown"
            improvement = "N/A"
        
        comparison_data.append({
            'Metric': metric,
            'Expected': expected_value,
            'Original': original_value,
            'Enhanced': enhanced_value,
            'Error Reduction': improvement,
            'Status': status
        })
    
    # Create results DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate summary stats
    total_metrics = len(comparison_data)
    improved = sum(1 for row in comparison_data if row['Status'] == "✅ Improved")
    same = sum(1 for row in comparison_data if row['Status'] == "🟰 Same")
    regression = sum(1 for row in comparison_data if row['Status'] == "❌ Regression")
    new = sum(1 for row in comparison_data if row['Status'] == "➕ New")
    unknown = sum(1 for row in comparison_data if row['Status'] == "❓ Unknown")
    
    improvement_rate = (improved / (improved + same + regression)) * 100 if (improved + same + regression) > 0 else 0
    
    return {
        'comparison_df': comparison_df,
        'total_metrics': total_metrics,
        'improved': improved,
        'same': same,
        'regression': regression,
        'new': new,
        'unknown': unknown,
        'improvement_rate': improvement_rate
    }

async def run_test():
    """Run the metrics calculators comparison test."""
    print("=== Enhanced Metrics Calculator Test ===")
    print("This test compares the original and enhanced metrics calculators against expected values")
    
    # Create pool
    pool = await asyncpg.create_pool(**DB_CONFIG)
    
    try:
        print("\n1. Setting up test data...")
        async with pool.acquire() as conn:
            # Set up test data and get expected metrics
            expected_metrics = await setup_test_data(conn)
            
            # Run original calculator
            print("\n2. Running original metrics calculator...")
            original_results = await run_original_calculator(pool, 9999, "TEST")
            
            # Run enhanced calculator
            print("\n3. Running enhanced metrics calculator...")
            enhanced_results = await run_enhanced_calculator(pool, 9999, "TEST")
            
            # Compare results
            print("\n4. Comparing results...")
            comparison = compare_calculators(expected_metrics, original_results, enhanced_results)
            
            # Print comparison table
            print("\n=== Metrics Comparison ===")
            print(tabulate.tabulate(comparison['comparison_df'], headers='keys', tablefmt='psql', showindex=False))
            
            # Print summary
            print("\n=== Improvement Summary ===")
            print(f"Total metrics compared:   {comparison['total_metrics']}")
            print(f"✅ Improved:              {comparison['improved']}")
            print(f"🟰 Same:                  {comparison['same']}")
            print(f"❌ Regression:            {comparison['regression']}")
            print(f"➕ New metrics:           {comparison['new']}")
            print(f"❓ Unknown comparison:    {comparison['unknown']}")
            
            if (comparison['improved'] + comparison['same'] + comparison['regression']) > 0:
                print(f"Improvement rate:        {comparison['improvement_rate']:.1f}%")
            
            # Save detailed results
            comparison['comparison_df'].to_csv("improved_metrics_results.csv", index=False)
            print("\nDetailed results saved to improved_metrics_results.csv")
            
            # Cleanup test data
            print("\n5. Cleaning up test data...")
            await conn.execute(f"""
                DELETE FROM sim_bot_trades WHERE bot_id = 9999;
                DELETE FROM sim_bots WHERE bot_id = 9999;
                DELETE FROM bot_metrics WHERE bot_id = 9999;
            """)
    
    except Exception as e:
        logger.error(f"Error during enhanced metrics test: {e}")
        print(f"\n❌ Error: {e}")
    
    finally:
        # Close pool
        await pool.close()
        print("\nTest completed.")

if __name__ == "__main__":
    # Verify dependencies
    missing_deps = []
    try:
        import asyncpg
    except ImportError:
        missing_deps.append("asyncpg")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import tabulate
    except ImportError:
        missing_deps.append("tabulate")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("❌ Missing dependencies. Please install:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        sys.exit(1)
    
    # Run the test
    asyncio.run(run_test())