import asyncpg
import math
from fastapi import FastAPI, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import logging
import os
from datetime import datetime

# --- Configuration ---
DATABASE_URL = "postgres://clayb:musicman@localhost:5432/tick_data"
# Allowed fields for sorting the bot list
ALLOWED_SORT_FIELDS = {
    "name": "b.name",
    "ticker": "b.ticker",
    "algorithm_type": "b.algorithm_type",
    "total_pnl": "m.total_pnl",
    "win_rate": "m.win_rate",
    "sharpe_ratio": "m.sharpe_ratio",
    "max_drawdown": "m.max_drawdown",
    "total_trades": "m.total_trades",
    "rank_score": "m.rank_score",
    "last_updated": "m.last_updated",
    "bot_id": "b.bot_id", # Allow sorting by id
}

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Pool ---
db_pool: asyncpg.Pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        logger.info("Creating database connection pool...")
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
            logger.info("Database connection pool created successfully.")
        except Exception as e:
            logger.exception(f"Failed to create database connection pool: {e}")
            raise HTTPException(status_code=503, detail="Database connection unavailable.")
    return db_pool

async def close_db_pool():
    global db_pool
    if db_pool:
        logger.info("Closing database connection pool...")
        await db_pool.close()
        db_pool = None
        logger.info("Database connection pool closed.")

# --- Pydantic Models ---

class BotSummary(BaseModel):
    bot_id: int
    name: str
    ticker: str
    algorithm_type: str
    is_active: bool
    total_pnl: Optional[float] = None
    win_rate: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    rank_score: Optional[float] = None
    last_updated: Optional[datetime] = None

class Pagination(BaseModel):
    current_page: int
    per_page: int
    total_items: int
    total_pages: int

class BotListResponse(BaseModel):
    data: List[BotSummary]
    pagination: Pagination

class BotDetail(BaseModel):
    # Fields from sim_bots
    bot_id: int
    name: str
    ticker: str
    algorithm_module: str
    algorithm_type: str
    trade_direction: str
    position_size: float
    trailing_stop_pct: float
    description: Optional[str] = None
    version: Optional[str] = None
    is_active: bool
    created_at: datetime
    # Fields from bot_metrics (potentially nullable if no metrics yet)
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    total_pnl: Optional[float] = None
    average_pnl_per_trade: Optional[float] = None
    win_rate: Optional[float] = None
    average_win_amount: Optional[float] = None
    average_loss_amount: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expectancy: Optional[float] = None
    rank_score: Optional[float] = None
    last_updated: Optional[datetime] = None

# --- Helper Functions ---

def build_list_query(
    ticker: Optional[str] = None,
    algorithm_type: Optional[str] = None,
    is_active: Optional[bool] = True, # Default to active bots
    updated_since: Optional[datetime] = None,
    sort_by: str = "rank_score",
    sort_order: str = "desc",
    limit: int = 25,
    offset: int = 0
) -> Tuple[str, str, List[Any]]:
    """Constructs the SELECT and COUNT queries with parameters."""

    base_query = """
        FROM sim_bots b
        LEFT JOIN bot_metrics m ON b.bot_id = m.bot_id
    """
    conditions = []
    params = []
    param_count = 1

    # --- Filtering ---
    if is_active is not None:
        conditions.append(f"b.is_active = ${param_count}")
        params.append(is_active)
        param_count += 1

    if ticker:
        conditions.append(f"b.ticker = ${param_count}")
        params.append(ticker)
        param_count += 1

    if algorithm_type:
        conditions.append(f"b.algorithm_type = ${param_count}")
        params.append(algorithm_type)
        param_count += 1

    if updated_since:
        conditions.append(f"m.last_updated > ${param_count}")
        params.append(updated_since)
        param_count += 1

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # --- Sorting ---
    sort_column = ALLOWED_SORT_FIELDS.get(sort_by, "m.rank_score") # Default sort
    order = "DESC" if sort_order.lower() == "desc" else "ASC" # Default DESC
    order_by_clause = f"ORDER BY {sort_column} {order}, b.bot_id ASC" # Secondary sort by ID for stability

    # --- Final Queries ---
    select_query = f"""
        SELECT
            b.bot_id, b.name, b.ticker, b.algorithm_type, b.is_active,
            m.total_pnl, m.win_rate, m.sharpe_ratio, m.max_drawdown,
            m.total_trades, m.rank_score, m.last_updated
        {base_query}
        {where_clause}
        {order_by_clause}
        LIMIT ${param_count} OFFSET ${param_count + 1}
    """
    params.extend([limit, offset])

    count_query = f"""
        SELECT COUNT(*)
        {base_query}
        {where_clause}
    """
    # Params for count query are the same as select query, minus limit/offset
    count_params = params[:-2]

    return select_query, count_query, count_params, params # Return params separately


# --- FastAPI App and Endpoints ---
app = FastAPI(title="Know-Defeat Metrics API")

@app.on_event("startup")
async def startup_event():
    await get_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    await close_db_pool()

@app.get("/api/metrics/bots", response_model=BotListResponse)
async def list_bots(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(25, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("rank_score", description=f"Field to sort by. Allowed values: {', '.join(ALLOWED_SORT_FIELDS.keys())}"),
    sort_order: str = Query("desc", description="Sort order ('asc' or 'desc')"),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    algorithm_type: Optional[str] = Query(None, description="Filter by algorithm type"),
    is_active: Optional[bool] = Query(True, description="Filter by active status (True, False, or omit for all)"),
    updated_since: Optional[datetime] = Query(None, description="Filter bots updated after this timestamp (ISO 8601 format)"),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Retrieves a paginated list of bots with key metrics, allowing filtering and sorting.
    """
    if sort_by not in ALLOWED_SORT_FIELDS:
        raise HTTPException(status_code=400, detail=f"Invalid sort_by field. Allowed values: {', '.join(ALLOWED_SORT_FIELDS.keys())}")
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Invalid sort_order. Use 'asc' or 'desc'.")

    offset = (page - 1) * per_page

    try:
        # Build queries and parameters
        select_query, count_query, count_params, select_params = build_list_query(
            ticker=ticker,
            algorithm_type=algorithm_type,
            is_active=is_active,
            updated_since=updated_since,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=per_page,
            offset=offset
        )

        # Execute queries concurrently (optional, but good practice)
        async with pool.acquire() as conn:
            total_items_task = conn.fetchval(count_query, *count_params)
            results_task = conn.fetch(select_query, *select_params)
            
            total_items = await total_items_task
            results = await results_task

        if total_items is None:
            total_items = 0

        # Format results
        bot_summaries = [BotSummary(**dict(record)) for record in results]

        total_pages = math.ceil(total_items / per_page) if total_items > 0 else 0

        pagination = Pagination(
            current_page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages
        )

        return BotListResponse(data=bot_summaries, pagination=pagination)

    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error fetching bot list: {e}")
        raise HTTPException(status_code=500, detail="Database query error.")
    except Exception as e:
        logger.exception(f"Unexpected error fetching bot list: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/api/metrics/bots/{bot_id}", response_model=BotDetail)
async def get_bot_details(
    bot_id: int = Path(..., ge=1, description="The unique identifier of the bot"),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Retrieves all configuration and metrics details for a specific bot.
    """
    query = """
        SELECT
            b.*, -- Select all columns from sim_bots
            m.*  -- Select all columns from bot_metrics
        FROM
            sim_bots b
        LEFT JOIN -- Use LEFT JOIN in case metrics don't exist yet
            bot_metrics m ON b.bot_id = m.bot_id
        WHERE
            b.bot_id = $1;
    """
    try:
        async with pool.acquire() as conn:
            record = await conn.fetchrow(query, bot_id)

        if record is None:
            raise HTTPException(status_code=404, detail=f"Bot with id {bot_id} not found.")

        # Convert asyncpg Record to dictionary for Pydantic validation
        bot_data = dict(record)

        # Pydantic automatically handles potential nulls from LEFT JOIN for metric fields
        return BotDetail(**bot_data)

    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database error fetching bot details for id {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Database query error.")
    except Exception as e:
        logger.exception(f"Unexpected error fetching bot details for id {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- Optional: Add main block to run with uvicorn for testing ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server for testing...")
    # Ensure environment variable is set if needed elsewhere, or use DATABASE_URL directly
    # os.environ['DATABASE_URL'] = DATABASE_URL
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info") 