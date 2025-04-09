@echo off
echo Starting Know Defeat UI with REAL database data...

:: Copy the real data environment file
copy .env.real .env

:: Start the development server
echo Starting development server with real database connection...
npm run dev