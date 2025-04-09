@echo off
echo Setting up Know Defeat Remix UI...

:: Remove previous installation
echo Cleaning previous installation...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del package-lock.json
if exist node-runner.mjs del node-runner.mjs 

:: Install dependencies
echo Installing dependencies with legacy-peer-deps...
npm install --legacy-peer-deps

echo Setup complete. Run start_ui.bat to start the development server.