import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_url():
    """Build database URL from environment variables"""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "embeddings")
    
    if not all([db_user, db_password, db_host]):
        raise ValueError("Missing required database environment variables: DB_USER, DB_PASSWORD, DB_HOST")
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def get_db_params():
    """Get individual database parameters"""
    return {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "embeddings")
    }