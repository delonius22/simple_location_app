"""Repository classes for SQLite database operations."""

import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional

from src.db.models import UserProfile, PatternAnalysis, LoginEvent

logger = logging.getLogger(__name__)

class UserProfileRepository:
    """Repository for user profile database operations."""
    
    @staticmethod
    def get_user_profile(db: Session, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id."""
        return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    @staticmethod
    def create_user_profile(db: Session, profile_data: Dict[str, Any]) -> UserProfile:
        """Create a new user profile."""
        try:
            # Convert JSON fields to strings for SQLite
            if 'devices' in profile_data and isinstance(profile_data['devices'], dict):
                profile_data['devices'] = json.dumps(profile_data['devices'])
            if 'locations' in profile_data and isinstance(profile_data['locations'], dict):
                profile_data['locations'] = json.dumps(profile_data['locations'])
            if 'time_patterns' in profile_data and isinstance(profile_data['time_patterns'], dict):
                profile_data['time_patterns'] = json.dumps(profile_data['time_patterns'])
            if 'behavioral_metrics' in profile_data and isinstance(profile_data['behavioral_metrics'], dict):
                profile_data['behavioral_metrics'] = json.dumps(profile_data['behavioral_metrics'])
            
            db_profile = UserProfile(**profile_data)
            db.add(db_profile)
            db.commit()
            db.refresh(db_profile)
            return db_profile
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating user profile: {str(e)}")
            raise
    
    @staticmethod
    def update_user_profile(db: Session, user_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        """Update an existing user profile."""
        try:
            db_profile = UserProfileRepository.get_user_profile(db, user_id)
            if not db_profile:
                return UserProfileRepository.create_user_profile(db, {**profile_data, 'user_id': user_id})
            
            # Convert JSON fields to strings for SQLite
            if 'devices' in profile_data and isinstance(profile_data['devices'], dict):
                profile_data['devices'] = json.dumps(profile_data['devices'])
            if 'locations' in profile_data and isinstance(profile_data['locations'], dict):
                profile_data['locations'] = json.dumps(profile_data['locations'])
            if 'time_patterns' in profile_data and isinstance(profile_data['time_patterns'], dict):
                profile_data['time_patterns'] = json.dumps(profile_data['time_patterns'])
            if 'behavioral_metrics' in profile_data and isinstance(profile_data['behavioral_metrics'], dict):
                profile_data['behavioral_metrics'] = json.dumps(profile_data['behavioral_metrics'])
            
            # Update fields
            for key, value in profile_data.items():
                setattr(db_profile, key, value)
            
            db_profile.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(db_profile)
            return db_profile
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating user profile: {str(e)}")
            raise

class PatternAnalysisRepository:
    """Repository for pattern analysis database operations."""
    
    @staticmethod
    def get_pattern_analysis(db: Session, user_id: str) -> Optional[PatternAnalysis]:
        """Get pattern analysis by user_id."""
        return db.query(PatternAnalysis).filter(PatternAnalysis.user_id == user_id).first()
    
    @staticmethod
    def update_pattern_analysis(db: Session, user_id: str, analysis_data: Dict[str, Any]) -> PatternAnalysis:
        """Create or update pattern analysis."""
        try:
            db_analysis = PatternAnalysisRepository.get_pattern_analysis(db, user_id)
            
            # Convert JSON fields to strings for SQLite
            if 'cluster_stats' in analysis_data and isinstance(analysis_data['cluster_stats'], dict):
                analysis_data['cluster_stats'] = json.dumps(analysis_data['cluster_stats'])
            if 'features' in analysis_data and isinstance(analysis_data['features'], list):
                analysis_data['features'] = json.dumps(analysis_data['features'])
            
            if not db_analysis:
                # Create new
                db_analysis = PatternAnalysis(user_id=user_id, **analysis_data)
                db.add(db_analysis)
            else:
                # Update existing
                for key, value in analysis_data.items():
                    setattr(db_analysis, key, value)
                db_analysis.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(db_analysis)
            return db_analysis
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating pattern analysis: {str(e)}")
            raise

class LoginEventRepository:
    """Repository for login event database operations."""
    
    @staticmethod
    def create_login_event(db: Session, login_data: Dict[str, Any], anomaly_result: Dict[str, Any] = None) -> LoginEvent:
        """Create a new login event record."""
        try:
            is_anomalous = False
            anomaly_score = 0.0
            details = None
            
            if anomaly_result:
                is_anomalous = anomaly_result.get('is_anomalous', False)
                anomaly_score = anomaly_result.get('combined_score', 0.0)
                # Convert details to JSON string for SQLite
                details = json.dumps(anomaly_result.get('details', {}))
            
            login_event = LoginEvent(
                user_id=login_data['user_id'],
                timestamp=login_data['timestamp'],
                latitude=login_data['latitude'],
                longitude=login_data['longitude'],
                device_id=login_data['device_id'],
                ip_address=login_data.get('ip_address'),
                is_anomalous=is_anomalous,
                anomaly_score=anomaly_score,
                details=details
            )
            
            db.add(login_event)
            db.commit()
            db.refresh(login_event)
            return login_event
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating login event: {str(e)}")
            raise
    
    @staticmethod
    def get_user_login_events(db: Session, user_id: str, limit: int = 1000) -> pd.DataFrame:
        """Get login events for a specific user."""
        events = db.query(LoginEvent).filter(
            LoginEvent.user_id == user_id
        ).order_by(LoginEvent.timestamp.desc()).limit(limit).all()
        
        # Convert to DataFrame for analysis
        if not events:
            return pd.DataFrame()
        
        data = []
        for event in events:
            data.append({
                'user_id': event.user_id,
                'timestamp': event.timestamp,
                'latitude': event.latitude,
                'longitude': event.longitude,
                'device_id': event.device_id,
                'ip_address': event.ip_address,
                'is_anomalous': event.is_anomalous,
                'anomaly_score': event.anomaly_score
            })
            
        return pd.DataFrame(data)