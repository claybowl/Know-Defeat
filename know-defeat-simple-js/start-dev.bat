@echo off
echo Starting development servers...

REM Start API server
echo Starting API server...
start cmd /k "cd api && npm run dev"

REM Wait for a moment to allow API to start
timeout /t 2 > nul

REM Start UI server
echo Starting UI development server...
start cmd /k "cd ui && npm install @rollup/rollup-win32-x64-msvc --no-save && npm run dev"

echo.
echo Development servers started:
echo API: http://localhost:8080
echo UI: http://localhost:5173
echo.
echo Close the command windows to stop the servers.