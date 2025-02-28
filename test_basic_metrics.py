#!/usr/bin/env python
# Simple test script to verify our type conversion functions

import math
from decimal import Decimal

def ensure_float(value, default=0.0):
    """Convert value to float, handling None, NaN, and infinity values safely."""
    if value is None:
        return default
    
    try:
        # Convert Decimal to float
        if isinstance(value, Decimal):
            value = float(value)
            
        # Handle float conversion
        result = float(value)
        
        # Check for NaN or infinite values
        if math.isnan(result) or math.isinf(result):
            return default
            
        return result
    except (ValueError, TypeError, OverflowError):
        return default

def test_convert_decimal_to_float():
    """Test converting Decimal to float."""
    print("\n--- Testing Decimal to Float Conversion ---")
    
    # Test cases
    test_cases = [
        {"input": Decimal("10.5"), "expected": 10.5, "name": "Simple Decimal"},
        {"input": Decimal("0"), "expected": 0.0, "name": "Zero Decimal"},
        {"input": Decimal("-10.5"), "expected": -10.5, "name": "Negative Decimal"},
        {"input": Decimal("9999999.9999"), "expected": 9999999.9999, "name": "Large Decimal"},
    ]
    
    for case in test_cases:
        result = ensure_float(case["input"])
        success = abs(result - case["expected"]) < 0.00001
        status = "PASS" if success else "FAIL"
        print(f"{status}: {case['name']} - Input: {case['input']}, Output: {result}, Expected: {case['expected']}")

def test_division_operations():
    """Test division operations with different types."""
    print("\n--- Testing Division Operations ---")
    
    # Test cases
    test_cases = [
        {"a": Decimal("10.5"), "b": 2.0, "expected": 5.25, "name": "Decimal / float"},
        {"a": 10.5, "b": Decimal("2.0"), "expected": 5.25, "name": "float / Decimal"},
        {"a": Decimal("10.5"), "b": Decimal("2.0"), "expected": 5.25, "name": "Decimal / Decimal"},
        {"a": 10.5, "b": 2.0, "expected": 5.25, "name": "float / float"},
        {"a": None, "b": 2.0, "expected": 0.0, "name": "None / float"},
        {"a": Decimal("10.5"), "b": None, "expected": 0.0, "name": "Decimal / None"},
    ]
    
    for case in test_cases:
        a_float = ensure_float(case["a"])
        b_float = ensure_float(case["b"])
        
        # Avoid division by zero
        if b_float == 0:
            result = 0.0
        else:
            result = a_float / b_float
            
        success = abs(result - case["expected"]) < 0.00001
        status = "PASS" if success else "FAIL"
        print(f"{status}: {case['name']} - Result: {result}, Expected: {case['expected']}")

def test_error_handling():
    """Test error handling in ensure_float."""
    print("\n--- Testing Error Handling ---")
    
    # Test cases
    test_cases = [
        {"input": "not a number", "default": 0.0, "expected": 0.0, "name": "String value"},
        {"input": None, "default": -1.0, "expected": -1.0, "name": "None with custom default"},
        {"input": float('inf'), "default": 999.0, "expected": 999.0, "name": "Infinity"},
        {"input": float('nan'), "default": 888.0, "expected": 888.0, "name": "NaN"},
        {"input": [1, 2, 3], "default": 777.0, "expected": 777.0, "name": "List"},
    ]
    
    for case in test_cases:
        result = ensure_float(case["input"], case["default"])
        
        # Special handling for NaN expected results
        if case["name"] == "NaN":
            success = (result == case["expected"])
        else:
            success = abs(result - case["expected"]) < 0.00001
            
        status = "PASS" if success else "FAIL"
        print(f"{status}: {case['name']} - Input: {case['input']}, Output: {result}, Expected: {case['expected']}")

def test_real_world_scenarios():
    """Test real-world scenarios that caused issues."""
    print("\n--- Testing Real-World Scenarios ---")
    
    # Scenario 1: Mixed decimal and float in win rate calculation
    winning_trades = Decimal("15")
    total_trades = 30.0
    
    winning_trades_float = ensure_float(winning_trades)
    total_trades_float = ensure_float(total_trades)
    
    win_rate = (winning_trades_float / total_trades_float) * 100
    expected = 50.0
    
    success = abs(win_rate - expected) < 0.00001
    status = "PASS" if success else "FAIL"
    print(f"{status}: Win Rate Calculation - Result: {win_rate}%, Expected: {expected}%")
    
    # Scenario 2: Drawdown calculation with mixed types
    pnl_values = [Decimal("10.5"), Decimal("-5.2"), 3.0, Decimal("8.1")]
    pnl_float_values = [ensure_float(pnl) for pnl in pnl_values]
    
    # Calculate running sum and drawdowns
    running_pnl = 0.0
    peak_pnl = 0.0
    drawdowns = []
    
    for pnl in pnl_float_values:
        running_pnl += pnl
        
        if running_pnl > peak_pnl:
            peak_pnl = running_pnl
        else:
            drawdown = peak_pnl - running_pnl
            drawdowns.append(drawdown)
    
    avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
    max_drawdown = max(drawdowns) if drawdowns else 0.0
    
    expected_avg = 2.6
    expected_max = 5.2
    
    success_avg = abs(avg_drawdown - expected_avg) < 0.00001
    success_max = abs(max_drawdown - expected_max) < 0.00001
    
    status_avg = "PASS" if success_avg else "FAIL"
    status_max = "PASS" if success_max else "FAIL"
    
    print(f"{status_avg}: Average Drawdown - Result: {avg_drawdown}, Expected: {expected_avg}")
    print(f"{status_max}: Maximum Drawdown - Result: {max_drawdown}, Expected: {expected_max}")

def run_all_tests():
    """Run all test functions."""
    print("===== METRICS CONVERSION TESTS =====")
    
    test_convert_decimal_to_float()
    test_division_operations()
    test_error_handling()
    test_real_world_scenarios()
    
    print("\n===== TEST SUMMARY =====")
    print("All tests completed. Check above for any FAIL statuses.")

if __name__ == "__main__":
    run_all_tests() 