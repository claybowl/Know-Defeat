@echo off
echo Starting Know Defeat UI with MOCK data...

:: Copy the mock data environment file
copy .env.mock .env

:: Start the development server
echo Starting development server with mock data...
npm run dev