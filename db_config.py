# db_config.py
import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_file():
    """Load .env file from frontend directory."""
    # Get the current file's directory
    current_dir = Path(__file__).resolve().parent
    # Look for frontend/.env
    env_path = current_dir / 'frontend' / '.env'
    
    if not env_path.exists():
        raise FileNotFoundError(
            f"Environment file not found at {env_path}. "
            f"Please create .env file in the frontend directory."
        )
    
    load_dotenv(env_path)

def get_db_url():
    """Get database URL from environment variables."""
    load_env_file()
    
    # Get database configuration from environment variables
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME')
    
    # Validate that required variables are present
    required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please check your .env file."
        )
    
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Add this for testing the configuration
if __name__ == "__main__":
    try:
        url = get_db_url()
        print("Successfully loaded database configuration!")
        print(f"Database host: {os.getenv('DB_HOST', 'localhost')}")
        print(f"Database name: {os.getenv('DB_NAME')}")
        print(f"Database user: {os.getenv('DB_USER')}")
    except Exception as e:
        print(f"Error: {str(e)}")