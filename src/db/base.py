"""Database connection and session management for SQLite."""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config import DATABASE_URL

# Configure logging
logger = logging.getLogger(__name__)

# Create SQLite engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created or verified")

def close_db_connections():
    """Close all database connections."""
    engine.dispose()
    logger.info("Database connections closed")