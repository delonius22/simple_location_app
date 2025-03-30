"""SQLAlchemy models for SQLite database."""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, 
    JSON, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship

from src.db.base import Base

class UserProfile(Base):
    """User profile database model."""
    
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    total_logins = Column(Integer, default=0)
    first_login = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)
    devices = Column(String, default="{}")  # JSON stored as string
    locations = Column(String, default="{}")  # JSON stored as string
    time_patterns = Column(String, default="{}")  # JSON stored as string
    behavioral_metrics = Column(String, default="{}")  # JSON stored as string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    login_events = relationship("LoginEvent", back_populates="user_profile")

class PatternAnalysis(Base):
    """Pattern analysis database model."""
    
    __tablename__ = "pattern_analyses"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("user_profiles.user_id", ondelete="CASCADE"), unique=True, index=True)
    cluster_stats = Column(String, default="{}")  # JSON stored as string
    num_clusters = Column(Integer, default=0)
    features = Column(String, default="[]")  # JSON stored as string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class LoginEvent(Base):
    """Login event database model."""
    
    __tablename__ = "login_events"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("user_profiles.user_id", ondelete="CASCADE"), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    device_id = Column(String, nullable=False)
    ip_address = Column(String, nullable=True)
    is_anomalous = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)
    details = Column(String, nullable=True)  # JSON stored as string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_profile = relationship("UserProfile", back_populates="login_events")