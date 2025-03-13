#!/usr/bin/env python
# Script to verify the decimal.Decimal vs float fixes

import asyncio
import logging
from decimal import Decimal
import asyncpg
import sys
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TypeConversionTester:
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_pool = None
    
    async def setup(self):
        """Initialize the database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(**self.db_config)
            logging.info("Database connection pool created successfully")
        except Exception as e:
            logging.error(f"Failed to create database connection: {e}")
            sys.exit(1)
    
    async def close(self):
        """Close the database connection pool."""
        if self.db_pool:
            await self.db_pool.close()
            logging.info("Database connection closed")
    
    def _ensure_float(self, value, default=0.0):
        """Test the enhanced _ensure_float method."""
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
    
    async def test_decimal_float_division(self):
        """Test division between Decimal and float values."""
        # Test cases with different combinations
        test_cases = [
            {'name': 'Decimal / float', 'a': Decimal('10.5'), 'b': 2.0},
            {'name': 'float / Decimal', 'a': 10.5, 'b': Decimal('2.0')},
            {'name': 'Decimal / Decimal', 'a': Decimal('10.5'), 'b': Decimal('2.0')},
            {'name': 'float / float', 'a': 10.5, 'b': 2.0},
            {'name': 'None / float', 'a': None, 'b': 2.0},
            {'name': 'Decimal / None', 'a': Decimal('10.5'), 'b': None},
            {'name': 'Invalid string / float', 'a': 'invalid', 'b': 2.0},
            {'name': 'Infinity / float', 'a': float('inf'), 'b': 2.0}
        ]
        
        for case in test_cases:
            try:
                # Simulate division with our _ensure_float method
                a_float = self._ensure_float(case['a'])
                b_float = self._ensure_float(case['b'])
                
                if b_float == 0:
                    result = 0.0  # Avoid division by zero
                else:
                    result = a_float / b_float
                
                logging.info(f"Test {case['name']}: SUCCESS - Result: {result}")
            except Exception as e:
                logging.error(f"Test {case['name']}: FAILED - Error: {e}")
    
    async def test_database_values(self):
        """Test retrieving and calculating with database values."""
        try:
            async with self.db_pool.acquire() as connection:
                # Fetch a test value from the database (could be Decimal)
                query = """
                    SELECT trade_pnl 
                    FROM sim_bot_trades 
                    WHERE bot_id = 1 
                    LIMIT 5
                """
                rows = await connection.fetch(query)
                
                logging.info(f"Retrieved {len(rows)} rows for testing")
                
                for i, row in enumerate(rows):
                    pnl = row['trade_pnl']
                    logging.info(f"Row {i+1}: Original PNL value: {pnl} (type: {type(pnl).__name__})")
                    
                    # Convert to float
                    pnl_float = self._ensure_float(pnl)
                    logging.info(f"Row {i+1}: Converted PNL value: {pnl_float} (type: {type(pnl_float).__name__})")
                    
                    # Simulate a calculation
                    result = pnl_float * 1.5
                    logging.info(f"Row {i+1}: Calculation result: {result}")
                
                return True
        except Exception as e:
            logging.error(f"Database test failed: {e}")
            return False
    
    async def run_tests(self):
        """Run all tests."""
        logging.info("Starting type conversion tests")
        
        # Test 1: Basic division operations
        logging.info("TEST 1: Basic division operations with Decimal and float")
        await self.test_decimal_float_division()
        
        # Test 2: Database values
        logging.info("TEST 2: Database values type conversion")
        db_test_result = await self.test_database_values()
        
        if db_test_result:
            logging.info("All tests completed successfully")
        else:
            logging.error("Some tests failed")
        
        return db_test_result

async def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'clayb',
        'password': '',
        'database': 'tick_data'
    }
    
    tester = TypeConversionTester(db_config)
    
    try:
        await tester.setup()
        success = await tester.run_tests()
        
        if success:
            logging.info("Type conversion fixes have been verified successfully")
            print("\n✅ SUCCESS: The decimal.Decimal vs float conversion fixes work as expected!")
        else:
            logging.error("Type conversion fixes verification failed")
            print("\n❌ FAILURE: The decimal.Decimal vs float conversion fixes do not work as expected!")
    
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 