@echo off
echo Starting Know Defeat Remix UI (fallback script)...

echo Approach 1: Using npm scripts...
call npm run dev

if %ERRORLEVEL% NEQ 0 (
  echo Approach 1 failed, trying approach 2: Direct execution...
  node remix-cli.js dev
)

if %ERRORLEVEL% NEQ 0 (
  echo Approach 2 failed, trying approach 3: Classic remix-dev...
  node remix-dev.js
)

if %ERRORLEVEL% NEQ 0 (
  echo All approaches failed.
  echo Please try running: npm install -g @remix-run/dev
  echo Then try again.
)