// Load environment variables from .env file
export function getEnv() {
  // Log for debugging env variables
  console.log(`Environment: NODE_ENV=${process.env.NODE_ENV}`);
  console.log(`DB Config: Host=${process.env.DB_HOST || 'localhost'}, DB=${process.env.DB_NAME || 'tick_data'}`);
  console.log(`Using Mock Data: ${process.env.USE_MOCK_DATA === 'true' ? 'YES' : 'NO'}`);
  
  return {
    // Database configuration
    DB_HOST: process.env.DB_HOST || 'localhost',
    DB_PORT: parseInt(process.env.DB_PORT || '5432'),
    DB_NAME: process.env.DB_NAME || 'tick_data',
    DB_USER: process.env.DB_USER || 'clayb',
    DB_PASSWORD: process.env.DB_PASSWORD || 'musicman',
    
    // Flag to use mock data instead of real database
    USE_MOCK_DATA: process.env.USE_MOCK_DATA === 'true',
  };
}

// Force use of real database (for debugging/testing)
export function forceRealDatabase() {
  console.log("⚠️ FORCING USE OF REAL DATABASE - Mock data disabled");
  process.env.USE_MOCK_DATA = 'false';
}

// Force use of mock data (for debugging/testing)
export function forceMockData() {
  console.log("⚠️ FORCING USE OF MOCK DATA - Real database disabled");
  process.env.USE_MOCK_DATA = 'true';
}

export default { getEnv, forceRealDatabase, forceMockData };