# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architect Instructions and Prime Directives

- Maintain consistency with the established code style and architectural patterns documented in this file and the `know-defeat-rules.mdc` guide.
- Proactively analyze the code context and anticipate potential issues or improvements.
- Don't just follow instructions literally; suggest better approaches or alternatives when appropriate.
- Explain the reasoning behind your suggestions and code changes clearly.
- Strive to understand the broader goals of the project and align your assistance accordingly.
- Identify and suggest refactoring opportunities to improve code quality, maintainability, and performance.
- When encountering errors or ambiguity, attempt to resolve them by referencing project documentation and context before asking for clarification.
- Prioritize solutions that are robust, efficient, and adhere to best practices.

## PostgreSQL Database Access via MCP

- **ALWAYS** use the MCP PostgreSQL tool for database operations when available
- Run SQL queries directly using the `mcp__postgres2__query` tool
- PostgreSQL connection is pre-configured to access the `tick_data` database
- For complex operations, prefer direct SQL over ORM abstractions
- Schema exploration: Query information_schema tables to understand database structure
- When modifying data, use transactions where appropriate
- Database tables: bot_metrics, bots, tick_data, bot_rankings, account_history, and others
- Example usage:
  ```sql
  -- Get all bots with their metrics
  SELECT b.*, m.* 
  FROM bots b
  LEFT JOIN bot_metrics m ON b.id = m.bot_id
  LIMIT 10;
  ```

## Build/Test Commands

- Python tests: `python -m pytest tests/` or single test `python -m pytest tests/test_bot_ranker.py::test_fetch_bot_metrics`
- Direct test file: `python tests/test_basic_metrics.py`
- Trading pipeline tests sequence:
  
  ``` python
  python tests/test_trade_creation.py
  python tests/test_metrics_system.py
  python tests/test_trading_pipeline.py
  ```

- Type checking: `mypy src/ algorithms/`
- Lint code: `flake8 src/ algorithms/ tests/`
- UI development: `npm run dev`
- UI type checking: `npm run typecheck`

## Code Style Guidelines

- **Imports**: Standard library first, then third-party, then local modules
- **Types**: Use type hints (Dict, List, Any, Optional) for function parameters and returns
- **Docstrings**: Google-style with Args/Returns sections
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error handling**: Use specific exception types with contextual logging
- **Async**: Use asyncio patterns with proper async/await and context managers
- **Database**: Use asyncpg for async database operations
- **Indentation**: 4 spaces for Python, 2 spaces for TypeScript/React code
- **UI Components**: Follow Chakra UI patterns, use TypeScript for all components

## Environment Setup

- Use Anaconda environment: `conda activate Autogen`
- PostgreSQL database:
  - Start: `pg_ctl -D "C:/Users/clayb/postgres_data" start`
  - Connect: `psql -U clayb -d tick_data`
- Store logs in `logs/` directory and documents in `docs/` directory
