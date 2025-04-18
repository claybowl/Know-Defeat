import asyncio
import logging
import signal
from typing import Dict, Any, List
import json
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Import application components
from src.api.monitoring_endpoints import router as monitoring_router
from src.performance_poller import PerformancePoller
from src.db.notifications_setup import setup_notifications
from src.db_connection import initialize_db_pool, close_db_pool, get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ]
)
logger = logging.getLogger("main")

# Create FastAPI app
app = FastAPI(
    title="Know-Defeat Trading System",
    description="Real-time monitoring dashboard for algorithmic trading",
    version="1.0.0"
)

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, limit this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
performance_poller: PerformancePoller = None
app_state: Dict[str, Any] = {
    "is_running": False,
    "active_connections": 0
}

# Mount the monitoring API endpoints
app.include_router(monitoring_router)

# Mount static files for the frontend
# Uncomment when you have a frontend build
# app.mount("/", StaticFiles(directory="ui/build", html=True), name="ui")

@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup."""
    global performance_poller
    logger.info("Starting Know-Defeat Trading System")
    
    # Initialize database pool using the new function
    await initialize_db_pool()
    
    # Set up database notifications
    logger.info("Setting up database notification triggers")
    await setup_notifications()
    
    # Start performance poller in the background
    logger.info("Starting performance polling system")
    performance_poller = PerformancePoller(poll_interval_seconds=120)  # Poll every 2 minutes
    asyncio.create_task(performance_poller.run())
    
    app_state["is_running"] = True
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global performance_poller
    logger.info("Shutting down Know-Defeat Trading System")
    
    # Stop performance poller
    if performance_poller:
        logger.info("Stopping performance poller")
        await performance_poller.stop()
    
    # Close database pool using the new function
    await close_db_pool()
    
    app_state["is_running"] = False
    logger.info("Application shutdown complete")

@app.get("/")
async def root():
    """Root endpoint that redirects to docs."""
    return {"message": "Know-Defeat Trading System API", "docs": "/docs"}

@app.get("/api/system/status")
async def system_status():
    """Get overall system status."""
    global performance_poller
    
    # Check DB status (can refine this later, maybe add a check_connection function)
    # For now, assume connected if initialized, but could be better
    from src.db_connection import pool as db_pool_global # Access the global pool
    db_status = "connected" if db_pool_global else "disconnected"
    
    # Check if performance poller is running
    poller_status = "running" if performance_poller and performance_poller.running else "stopped"
    
    return {
        "system": {
            "status": "online" if app_state["is_running"] else "offline",
            "database": db_status,
            "poller": poller_status,
            "active_connections": app_state["active_connections"]
        }
    }

@app.middleware("http")
async def track_connections(request: Request, call_next):
    """Middleware to track active connections."""
    app_state["active_connections"] += 1
    try:
        response = await call_next(request)
        return response
    finally:
        app_state["active_connections"] -= 1

def start():
    """Entry point for running the application."""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start() 