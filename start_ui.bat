@echo off
echo Starting Know Defeat UI...

:: Check if node_modules exists, if not install dependencies
if not exist node_modules (
  echo Installing dependencies...
  npm install
)

:: Start the development server
echo Starting development server...
npm run dev