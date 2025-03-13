# Know-Defeat Algorithmic Trading System Guide

## Build/Test Commands
- Run all tests: `python -m pytest tests/`
- Run specific test: `python -m pytest tests/test_bot_ranker.py::test_fetch_bot_metrics`
- Run direct test file: `python tests/test_basic_metrics.py`
- Run Streamlit UI: `streamlit run user_interface/src/streamlit_app2.py`
- Export trades: `python user_interface/src/export_all_trades.py`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local modules
- **Types**: Use type hints (Dict, List, Any, Optional) for arguments and returns
- **Docstrings**: Google-style with Args/Returns sections
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error handling**: Use specific exception types with contextual logging
- **Logging**: Appropriate levels (debug, info, warning, error) with context
- **Async**: Use asyncio patterns with proper async/await and context managers
- **Database**: Use asyncpg for async database operations
- **Indentation**: 4 spaces, consistent throughout codebase

## Project Structure
- `algorithms/`: Trading algorithm implementations
- `src/`: Core system components and utilities
- `tests/`: Test suite using pytest
- `user_interface/`: Streamlit-based UI components