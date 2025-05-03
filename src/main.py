import asyncio
import logging
import signal
from typing import Dict, Any, List
import json
import os
import mimetypes
import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
import httpx

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
    allow_origins=["*"],  # For production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers to the browser
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Global state
performance_poller: PerformancePoller = None
app_state: Dict[str, Any] = {
    "is_running": False,
    "active_connections": 0
}

# Ensure JavaScript files are served with the correct MIME type
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')
mimetypes.add_type('application/json', '.map')

# Debug middleware
@app.middleware("http")
async def debug_request(request: Request, call_next):
    """Debug middleware to log request information."""
    # Log request details
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Client host: {request.client.host if request.client else 'unknown'}")
    
    # Call the next middleware/route handler
    response = await call_next(request)
    
    # Log response details
    logger.info(f"Response status: {response.status_code}")
    
    return response

# Mount the monitoring API endpoints
app.include_router(monitoring_router, prefix="/api")

# Define the location of your static files
BUILD_DIR = os.path.join(os.getcwd(), "build")

# Main middleware to handle all non-API requests by proxying to the dev server
@app.middleware("http")
async def proxy_to_dev_server(request: Request, call_next):
    # If it's an API request, let it go through to our API endpoints
    if request.url.path.startswith("/api"):
        return await call_next(request)
        
    # For all other paths, proxy to React dev server
    try:
        target_url = f"http://localhost:3000{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        logger.info(f"Proxying to dev server: {target_url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ("host",)},
                content=await request.body(),
                follow_redirects=True
            )
            
            logger.info(f"Dev server response: {response.status_code}")
            
            # Return the proxied response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type", "text/html")
            )
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        # If dev server is unavailable, try serving static files as fallback
        return await call_next(request)

# Fallback route for static file serving
@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(BUILD_DIR, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Default to index.html for SPA
    index_path = os.path.join(BUILD_DIR, "index.html")
    if os.path.exists(index_path) and os.path.isfile(index_path):
        return FileResponse(index_path)
        
    # Last resort
    return HTMLResponse(
        "<html><body><h1>Error</h1><p>Application not found. Make sure the React dev server is running.</p></body></html>",
        status_code=404
    )

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