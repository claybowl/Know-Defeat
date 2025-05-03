# Remote Access Guide for Know-Defeat Dashboard

This guide documents the process of making the Know-Defeat Trading Dashboard accessible remotely via ngrok.

## Overview

The deployment strategy uses:
1. FastAPI backend serving both API endpoints and static assets
2. React frontend for the user interface
3. Ngrok for tunneling to make the local server accessible remotely

## Prerequisites

- FastAPI and React applications set up and running locally
- Python 3.8+ with conda environment `Autogen`
- Node.js for the React frontend
- Ngrok account (free tier works)

## Setup Process

### 1. Install Required Dependencies

```bash
# Activate conda environment
conda activate Autogen

# Install required Python packages
pip install httpx
```

### 2. Configure FastAPI to Serve Frontend

FastAPI needs to be configured to handle both API requests and frontend rendering. We use a hybrid approach:

- API endpoints are mounted at `/api/*`
- Frontend requests are proxied to the React dev server during development
- Static files are served directly from the build directory as a fallback

Update `src/main.py` to implement this approach:

```python
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response

# Create FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API endpoints
app.include_router(api_router, prefix="/api")

# Define build directory
BUILD_DIR = os.path.join(os.getcwd(), "build")

# Serve static assets
app.mount("/index.js", StaticFiles(directory=BUILD_DIR, html=False), name="js-file")
app.mount("/index.js.map", StaticFiles(directory=BUILD_DIR, html=False), name="js-map-file")

# Proxy middleware
@app.middleware("http")
async def proxy_frontend(request: Request, call_next):
    # Let API requests go through to FastAPI
    if request.url.path.startswith("/api"):
        return await call_next(request)
    
    # Check for specific static files
    if request.url.path == "/index.js" or request.url.path == "/index.js.map":
        return await call_next(request)
    
    # For everything else, proxy to React dev server
    try:
        target_url = f"http://localhost:3000{request.url.path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers={k: v for k, v in request.headers.items() if k.lower() not in ("host",)},
                content=await request.body(),
                follow_redirects=True
            )
            
            # Handle 404 from dev server by trying to serve from build
            if response.status_code == 404 and os.path.exists(BUILD_DIR):
                file_path = os.path.join(BUILD_DIR, request.url.path.lstrip('/'))
                if os.path.exists(file_path):
                    return FileResponse(file_path)
                    
                index_path = os.path.join(BUILD_DIR, "index.html")
                if os.path.exists(index_path):
                    return FileResponse(index_path)
            
            # Return the proxied content
            content_type = response.headers.get("content-type", "text/html")
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type
            )
    except Exception as e:
        # If proxy fails, check if we can serve from build directory
        file_path = os.path.join(BUILD_DIR, request.url.path.lstrip('/'))
        if os.path.exists(file_path):
            return FileResponse(file_path)
            
        index_path = os.path.join(BUILD_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
            
        return HTMLResponse(
            "<html><body><h1>Error</h1><p>Cannot connect to development server.</p></body></html>",
            status_code=500
        )
```

### 3. Create HTML Template

If you're using a JavaScript-only build (no HTML), create an `index.html` file in your build directory:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Know-Defeat Trading Dashboard</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            height: 100%;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        #root {
            height: 100%;
            width: 100%;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script src="/index.js"></script>
</body>
</html>
```

### 4. Run the Development Environment

To start the development environment:

```bash
# Terminal 1: Start React development server
cd user_interface
npm run dev

# Terminal 2: Start FastAPI server
conda activate Autogen
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Set Up Ngrok for Remote Access

Download and install ngrok from the [official website](https://ngrok.com/download) instead of using npm:

```bash
# Navigate to the location where you extracted ngrok.exe
cd C:\path\to\ngrok

# Add your authtoken
ngrok.exe config add-authtoken YOUR_AUTHTOKEN

# Create a tunnel to your FastAPI server
ngrok.exe http 8000
```

After running ngrok, you'll see output similar to:

```
Forwarding https://your-subdomain.ngrok.io -> http://localhost:8000
```

Share this URL with your client to access the dashboard remotely.

## Troubleshooting

### Common Issues

1. **White Screen / No Content**
   - Check browser console for JavaScript errors
   - Make sure React dev server is running
   - Verify your index.html is properly loading index.js

2. **"Not Found" Error**
   - Check API routes are correctly prefixed with "/api"
   - Verify that static files exist in the expected directories
   - Look for paths that might be hardcoded in the React app

3. **Ngrok Connection Issues**
   - Use the standalone ngrok.exe rather than npm-installed version
   - Run in Windows Command Prompt or PowerShell if Git Bash fails
   - Verify your authtoken is correctly configured

### Checking Logs

- FastAPI logs: Check the uvicorn console and `logs/app.log`
- React logs: Check the React development server console
- Ngrok logs: Check the ngrok console output

## Production Deployment Recommendations

For a more permanent production setup:

1. **Build the Frontend Properly**
   ```bash
   cd user_interface
   npm run build
   ```

2. **Use a Reverse Proxy**
   Consider setting up Nginx or Caddy as a reverse proxy instead of ngrok

3. **Add HTTPS**
   Obtain an SSL certificate from Let's Encrypt for proper HTTPS

4. **Use a Process Manager**
   Set up a process manager like systemd or supervisor to keep the server running

5. **Consider a VPS**
   For permanent hosting, deploy to a Virtual Private Server (VPS)

## Alternative to Ngrok

For a more permanent solution, consider:

1. **Port forwarding** on your router (security implications)
2. **VPS hosting** on services like Digital Ocean, AWS, or Azure
3. **Cloudflare Tunnel** as a more secure alternative to ngrok

## Conclusion

This setup allows you to quickly share your dashboard with clients while maintaining the ability to use your development environment for real-time updates. For long-term production use, consider the recommendations above for a more robust solution. 