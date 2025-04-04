import asyncpg
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def create_db_pool():
    """
    Create a connection pool to the GCP Cloud SQL database.
    Uses environment variables from .env file.
    
    Returns:
        asyncpg.Pool: A connection pool to the database
    """
    return await asyncpg.create_pool(
        user=os.environ.get('DB_USER', 'postgres'),
        password=os.environ.get('DB_PASSWORD', 'musicman'),
        database=os.environ.get('DB_NAME', 'tick_data'),
        host=os.environ.get('DB_HOST', '127.0.0.1'),
        port=int(os.environ.get('DB_PORT', '5432')),
        # Recommended settings for Cloud SQL
        min_size=5,
        max_size=20,
        command_timeout=30,
        max_inactive_connection_lifetime=300
    )

# For connection to Cloud SQL using Unix socket (for Cloud Run, App Engine, etc.)
async def create_cloud_socket_pool():
    """
    Create a connection pool to Cloud SQL using Unix socket.
    This is for direct connection from GCP services like Cloud Run or App Engine.
    
    Returns:
        asyncpg.Pool: A connection pool to the database
    """
    connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_password = os.environ.get('DB_PASSWORD', 'musicman')
    db_name = os.environ.get('DB_NAME', 'tick_data')
    
    # Format: postgres://{db_user}:{db_pass}@/{db_name}?host=/cloudsql/{connection_name}
    return await asyncpg.create_pool(
        dsn=f"postgres://{db_user}:{db_password}@/{db_name}?host=/cloudsql/{connection_name}",
        min_size=5,
        max_size=20,
        command_timeout=30,
        max_inactive_connection_lifetime=300
    )