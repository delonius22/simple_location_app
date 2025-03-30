"""User login processor for anomaly detection."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
from scipy import stats
import json

from src.config import CACHE_MAX_SIZE, DEFAULT_ANOMALY_THRESHOLD
from src.db.repositories import (
    UserProfileRepository,
    PatternAnalysisRepository,
    LoginEventRepository
)
from src.services.geo_utils import (
    handle_antimeridian,
    calculate_geographic_mean_center,
    calculate_standard_distance,
    haversine_distance
)

logger = logging.getLogger(__name__)


class UserLoginProcessor:
    """Main class for processing and analyzing user login data with database integration."""
    
    def __init__(self, db: Session = None):
        """Initialize the login processor.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        
        # In-memory cache for frequently accessed profiles
        self.profile_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = CACHE_MAX_SIZE
    
    
    # Data Loading Methods
  
    
    def _load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        
        # Ensure user_id is a hashable type (string)
        if isinstance(user_id, pd.Series):
            user_id = user_id.iloc[0] if not user_id.empty else None
            if user_id is None:
                return None
            user_id = str(user_id)
            
        # Check cache first
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]
        
        # Get from database
        profile = UserProfileRepository.get_user_profile(self.db, user_id)
        if not profile:
            return None
        
        # Convert to dictionary for consistency - parse JSON strings from SQLite
        profile_dict = {
            'user_id': profile.user_id,
            'total_logins': profile.total_logins,
            'first_login': profile.first_login,
            'last_login': profile.last_login,
            'devices': json.loads(profile.devices) if profile.devices else {},
            'locations': json.loads(profile.locations) if profile.locations else {},
            'time_patterns': json.loads(profile.time_patterns) if profile.time_patterns else {},
            'behavioral_metrics': json.loads(profile.behavioral_metrics) if profile.behavioral_metrics else {}
        }
        
        # Update cache
        self.profile_cache[user_id] = profile_dict
        return profile_dict

    def _load_pattern_analysis(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load pattern analysis from database or cache.
        
        Args:
            user_id: User identifier
            
        Returns:
            Pattern analysis dictionary or None if not found
        """
        # Ensure user_id is a hashable type (string)
        if isinstance(user_id, pd.Series):
            user_id = user_id.iloc[0] if not user_id.empty else None
            if user_id is None:
                return None
            user_id = str(user_id)
            
        # Check cache first
        if user_id in self.pattern_cache:
            return self.pattern_cache[user_id]
        
        # Get from database
        analysis = PatternAnalysisRepository.get_pattern_analysis(self.db, user_id)
        if not analysis:
            return None
        
        # Convert to dictionary for consistency
        analysis_dict = {
            'user_id': analysis.user_id,
            'cluster_stats': analysis.cluster_stats,
            'num_clusters': analysis.num_clusters,
            'features': analysis.features
        }
        
        # Get user login data for user_data_with_clusters (needed for some algorithms)
        df = LoginEventRepository.get_user_login_events(self.db, user_id)
        if not df.empty:
            # We should re-run clustering to get cluster labels
            # This is a simplified approach - in production you'd want to store cluster assignments
            analysis_dict['user_data_with_clusters'] = df
        
        # Update cache with LRU policy
        if len(self.pattern_cache) >= self.max_cache_size:
            # Remove oldest item
            self.pattern_cache.pop(next(iter(self.pattern_cache)))
        
        self.pattern_cache[user_id] = analysis_dict
        return analysis_dict
    
    def _save_user_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """Save a user profile to database.
        
        Args:
            user_id: User identifier
            profile: User profile dictionary
        """
        # Update cache
        self.profile_cache[user_id] = profile
        
        # Save to database
        UserProfileRepository.update_user_profile(self.db, user_id, profile)
    
    def _save_pattern_analysis(self, user_id: str, analysis: Dict[str, Any]) -> None:
        """Save pattern analysis to database.
        
        Args:
            user_id: User identifier
            analysis: Pattern analysis dictionary
        """
        # Update cache
        self.pattern_cache[user_id] = analysis
        
        # Remove dataframe before saving to database
        db_analysis = analysis.copy()
        if 'user_data_with_clusters' in db_analysis:
            del db_analysis['user_data_with_clusters']
        
        # Save to database
        PatternAnalysisRepository.update_pattern_analysis(self.db, user_id, db_analysis)
    

    # Core Processing Methods
  
    
    def process_login_data(self, login_data: pd.DataFrame) -> Dict[str, Any]:
        """Process login data to create user profiles and analyze patterns.
        
        Args:
            login_data: DataFrame containing login records
                Required columns: user_id, timestamp, latitude, longitude, device_id
                
        Returns:
            Dictionary with processing statistics
        """
        # Ensure required columns exist
        required_columns = ['user_id', 'timestamp', 'latitude', 'longitude', 'device_id']
        for col in required_columns:
            if col not in login_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(login_data['timestamp']):
            login_data['timestamp'] = pd.to_datetime(login_data['timestamp'])
        
        # Process each user's data
        users_processed = 0
        new_users = 0
        updated_users = 0
        
        # Group by user_id for efficient processing
        user_groups = login_data.groupby('user_id')
        
        for user_id, user_data in user_groups:
            # Store login events in the database
            for _, row in user_data.iterrows():
                login_info = {
                    'user_id': row['user_id'],
                    'timestamp': row['timestamp'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'device_id': row['device_id'],
                    'ip_address': row.get('ip_address')
                }
                LoginEventRepository.create_login_event(self.db, login_info)
            
            # Create or update user profile
            profile = self._load_user_profile(user_id)
            if not profile:
                new_users += 1
                profile = self._create_user_profile(user_id, user_data)
            else:
                updated_users += 1
                # Update profile with new data
                profile = self._update_user_profile(profile, user_data)
            
            self._save_user_profile(user_id, profile)
            
            # Analyze user patterns
            pattern_analysis = self._identify_user_patterns(user_id, user_data)
            self._save_pattern_analysis(user_id, pattern_analysis)
            
            users_processed += 1
        
        return {
            'users_processed': users_processed,
            'new_users': new_users,
            'updated_users': updated_users
        }
    
    def _create_user_profile(
        self, 
        user_id: str, 
        login_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create a comprehensive user profile based on historical login data.
        
        Args:
            user_id: The unique identifier for the user
            login_data: DataFrame containing the user's login history
            
        Returns:
            A dictionary containing the user's profile information
        """
        profile = {
            'user_id': user_id,
            'total_logins': len(login_data),
            'first_login': login_data['timestamp'].min(),
            'last_login': login_data['timestamp'].max(),
            'devices': {},
            'locations': {},
            'time_patterns': {},
            'behavioral_metrics': {}
        }
        
        # Process devices
        devices = login_data['device_id'].value_counts().to_dict()
        for device_id, count in devices.items():
            profile['devices'][device_id] = {
                'count': count,
                'percentage': count / profile['total_logins'] * 100,
                'first_seen': login_data[login_data['device_id'] == device_id]['timestamp'].min().isoformat(),
                'last_seen': login_data[login_data['device_id'] == device_id]['timestamp'].max().isoformat()
            }
        
        # Process geographic data - use a grid approach for efficiency
        for idx, row in login_data.iterrows():
            # Round location for binning
            lat_rounded = round(row['latitude'], 3)
            lon_rounded = round(row['longitude'], 3)
            location_key = f"{lat_rounded}_{lon_rounded}"
            
            if location_key in profile['locations']:
                profile['locations'][location_key]['count'] += 1
                profile['locations'][location_key]['last_seen'] = row['timestamp'].isoformat()
            else:
                profile['locations'][location_key] = {
                    'count': 1,
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'first_seen': row['timestamp'].isoformat(),
                    'last_seen': row['timestamp'].isoformat()
                }
        
        # Add time pattern analysis
        login_data['hour'] = login_data['timestamp'].dt.hour
        login_data['day_of_week'] = login_data['timestamp'].dt.dayofweek
        
        # Hour distribution
        hour_counts = login_data['hour'].value_counts().sort_index()
        profile['time_patterns']['hour_distribution'] = hour_counts.to_dict()
        
        # Day of week distribution
        dow_counts = login_data['day_of_week'].value_counts().sort_index()
        profile['time_patterns']['day_of_week_distribution'] = dow_counts.to_dict()
        
        # Calculate login velocity
        profile['behavioral_metrics']['login_velocity'] = self._analyze_login_velocity(login_data)
        
        # Calculate geographical variance
        profile['behavioral_metrics']['geo_variance'] = self._calculate_combined_variance(login_data)
        
        return profile
    
    def _update_user_profile(
        self, 
        existing_profile: Dict[str, Any], 
        new_login_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Update an existing user profile with new login data.
        
        Args:
            existing_profile: The existing user profile dictionary
            new_login_data: DataFrame with new login data
            
        Returns:
            Updated profile dictionary
        """
        profile = existing_profile.copy()
        user_id = profile['user_id']
        
        # Update login counts and timestamps
        profile['total_logins'] += len(new_login_data)
        
        # Ensure consistent timezone handling for first_login comparison
        first_login = profile['first_login']
        new_min_timestamp = new_login_data['timestamp'].min()
        if first_login and hasattr(first_login, 'tzinfo') and first_login.tzinfo is not None:
            first_login = first_login.replace(tzinfo=None)
        if hasattr(new_min_timestamp, 'tzinfo') and new_min_timestamp.tzinfo is not None:
            new_min_timestamp = new_min_timestamp.replace(tzinfo=None)
        profile['first_login'] = min(first_login, new_min_timestamp) if first_login else new_min_timestamp
        
        # Ensure consistent timezone handling for last_login comparison
        last_login = profile['last_login']
        new_max_timestamp = new_login_data['timestamp'].max()
        if last_login and hasattr(last_login, 'tzinfo') and last_login.tzinfo is not None:
            last_login = last_login.replace(tzinfo=None)
        if hasattr(new_max_timestamp, 'tzinfo') and new_max_timestamp.tzinfo is not None:
            new_max_timestamp = new_max_timestamp.replace(tzinfo=None)
        profile['last_login'] = max(last_login, new_max_timestamp) if last_login else new_max_timestamp
        
        # Update device information
        for device_id, device_data in new_login_data.groupby('device_id'):
            if device_id in profile['devices']:
                # Update existing device
                device = profile['devices'][device_id]
                device['count'] += len(device_data)
                device['last_seen'] = max(
                    device['last_seen'],
                    device_data['timestamp'].max().isoformat()
                )
            else:
                # Add new device
                profile['devices'][device_id] = {
                    'count': len(device_data),
                    'percentage': 0, 
                    'first_seen': device_data['timestamp'].min().isoformat(),
                    'last_seen': device_data['timestamp'].max().isoformat()
                }
        
        # Update device percentages
        for device_id in profile['devices']:
            profile['devices'][device_id]['percentage'] = (
                profile['devices'][device_id]['count'] / profile['total_logins'] * 100
            )
        
        # Update location information
        for idx, row in new_login_data.iterrows():
            lat_rounded = round(row['latitude'], 3)
            lon_rounded = round(row['longitude'], 3)
            location_key = f"{lat_rounded}_{lon_rounded}"
            
            if location_key in profile['locations']:
                # Update existing location
                location = profile['locations'][location_key]
                location['count'] += 1
                location['last_seen'] = max(
                    location['last_seen'],
                    row['timestamp'].isoformat()
                )
            else:
                # Add new location
                profile['locations'][location_key] = {
                    'count': 1,
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'first_seen': row['timestamp'].isoformat(),
                    'last_seen': row['timestamp'].isoformat()
                }
        
        # Update time patterns
        new_login_data['hour'] = new_login_data['timestamp'].dt.hour
        new_login_data['day_of_week'] = new_login_data['timestamp'].dt.dayofweek
        
        # Update hour distribution
        hour_dist = profile['time_patterns']['hour_distribution']
        for hour, count in new_login_data['hour'].value_counts().to_dict().items():
            hour_key = str(hour)
            hour_dist[hour_key] = hour_dist.get(hour_key, 0) + count
        
        # Update day of week distribution
        day_dist = profile['time_patterns']['day_of_week_distribution']
        for day, count in new_login_data['day_of_week'].value_counts().to_dict().items():
            day_key = str(day)
            day_dist[day_key] = day_dist.get(day_key, 0) + count
        
        # Get all historical login data to recalculate metrics
        all_login_data = LoginEventRepository.get_user_login_events(self.db, user_id)
        if not all_login_data.empty:
            # Recalculate behavioral metrics with all data
            profile['behavioral_metrics']['login_velocity'] = self._analyze_login_velocity(all_login_data)
            profile['behavioral_metrics']['geo_variance'] = self._calculate_combined_variance(all_login_data)
        
        return profile
    
    def analyze_login(self, login_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single login attempt for anomalies.
        
        Args:
            login_info: Dictionary with login information
                
        Returns:
            Dictionary with anomaly scores and detailed analysis
        """
        user_id = login_info['user_id']
        
        # Check if we have data for this user
        user_profile = self._load_user_profile(user_id)
        pattern_analysis = self._load_pattern_analysis(user_id)
        
        if not user_profile or not pattern_analysis:
            # New user or insufficient data
            return {
                'user_id': user_id,
                'timestamp': login_info['timestamp'],
                'is_anomalous': False,  # Default for new users
                'combined_score': 0.5,  # Neutral score
                'anomaly_scores': {
                    'location': 0.5,
                    'device': 0.5,
                    'time': 0.5,
                    'velocity': 0.5,
                    'isolation_forest': 0.5,
                    'mahalanobis': 0.5
                },
                'threshold': DEFAULT_ANOMALY_THRESHOLD,
                'details': {
                    'location_details': {'message': 'Insufficient data'},
                    'device_details': {'message': 'Insufficient data'},
                    'time_details': {'message': 'Insufficient data'},
                    'velocity_details': {'message': 'Insufficient data'},
                    'isolation_forest_details': {'message': 'Insufficient data'},
                    'mahalanobis_details': {'message': 'Insufficient data'}
                }
            }
        
        # Calculate individual anomaly scores
        anomaly_scores = {}
        
        # 1. Location-based anomaly detection
        location_anomaly = self._detect_location_anomaly(login_info, pattern_analysis)
        anomaly_scores['location'] = location_anomaly['score']
        
        # 2. Device-based anomaly detection
        device_anomaly = self._detect_device_anomaly(login_info, user_profile)
        anomaly_scores['device'] = device_anomaly['score']
        
        # 3. Time-based anomaly detection
        time_anomaly = self._detect_time_anomaly(login_info, user_profile)
        anomaly_scores['time'] = time_anomaly['score']
        
        # 4. Velocity-based anomaly detection (impossible travel)
        velocity_anomaly = self._detect_velocity_anomaly(login_info, user_profile)
        anomaly_scores['velocity'] = velocity_anomaly['score']
        
        # Conditionally use advanced methods if we have enough data
        if 'user_data_with_clusters' in pattern_analysis and len(pattern_analysis['user_data_with_clusters']) >= 10:
            # 5. Isolation Forest for combined features
            if_anomaly = self._detect_anomalies_with_isolation_forest(login_info, pattern_analysis['user_data_with_clusters'])
            anomaly_scores['isolation_forest'] = if_anomaly['score']
            
            # 6. Mahalanobis distance for multivariate outlier detection
            mahalanobis_anomaly = self._calculate_mahalanobis_anomaly(login_info, pattern_analysis['user_data_with_clusters'])
            anomaly_scores['mahalanobis'] = mahalanobis_anomaly['score']
        else:
            # Not enough data for advanced methods
            anomaly_scores['isolation_forest'] = 0.5
            anomaly_scores['mahalanobis'] = 0.5
            if_anomaly = {'score': 0.5, 'details': {'message': 'Insufficient data'}}
            mahalanobis_anomaly = {'score': 0.5, 'details': {'message': 'Insufficient data'}}
        
        # Combine anomaly scores (weighted average)
        weights = {
            'location': 0.25,
            'device': 0.2,
            'time': 0.15,
            'velocity': 0.2,
            'isolation_forest': 0.1,
            'mahalanobis': 0.1
        }
        
        combined_score = sum(score * weights[key] for key, score in anomaly_scores.items())
        
        # Determine if this login is anomalous using a configurable threshold
        threshold = DEFAULT_ANOMALY_THRESHOLD
        is_anomalous = combined_score > threshold
        
        result = {
            'user_id': user_id,
            'timestamp': login_info['timestamp'],
            'is_anomalous': is_anomalous,
            'combined_score': combined_score,
            'anomaly_scores': anomaly_scores,
            'threshold': threshold,
            'details': {
                'location_details': location_anomaly['details'],
                'device_details': device_anomaly['details'],
                'time_details': time_anomaly['details'],
                'velocity_details': velocity_anomaly['details'],
                'isolation_forest_details': if_anomaly['details'],
                'mahalanobis_details': mahalanobis_anomaly['details']
            }
        }
        
        # Store login event with analysis result
        LoginEventRepository.create_login_event(self.db, login_info, result)
        
        # Update user profile with this login
        self.update_user_profile(login_info, result)
        
        return result
    
    def update_user_profile(
        self, 
        login_info: Dict[str, Any], 
        analysis_result: Dict[str, Any]
    ) -> None:
        """Update a user's profile after a login attempt (whether anomalous or not).
        
        Args:
            login_info: Dictionary with login information
            analysis_result: The result of analyze_login()
        """
        user_id = login_info['user_id']
        
        # Load existing profile or create a new one
        profile = self._load_user_profile(user_id)
        if not profile:
            # Create a minimal profile for new user
            profile = {
                'user_id': user_id,
                'total_logins': 1,
                'first_login': login_info['timestamp'],
                'last_login': login_info['timestamp'],
                'devices': {
                    login_info['device_id']: {
                        'count': 1,
                        'percentage': 100.0,
                        'first_seen': login_info['timestamp'].isoformat(),
                        'last_seen': login_info['timestamp'].isoformat()
                    }
                },
                'locations': {
                    f"{round(login_info['latitude'], 3)}_{round(login_info['longitude'], 3)}": {
                        'count': 1,
                        'lat': login_info['latitude'],
                        'lon': login_info['longitude'],
                        'first_seen': login_info['timestamp'].isoformat(),
                        'last_seen': login_info['timestamp'].isoformat()
                    }
                },
                'time_patterns': {
                    'hour_distribution': {str(login_info['timestamp'].hour): 1},
                    'day_of_week_distribution': {str(login_info['timestamp'].dayofweek): 1}
                },
                'behavioral_metrics': {}
            }
        else:
            # Update existing profile
            profile['total_logins'] += 1
            profile['last_login'] = max(profile['last_login'], login_info['timestamp'])
            
            # Update device info
            device_id = login_info['device_id']
            if device_id in profile['devices']:
                profile['devices'][device_id]['count'] += 1
                profile['devices'][device_id]['last_seen'] = login_info['timestamp'].isoformat()
            else:
                profile['devices'][device_id] = {
                    'count': 1,
                    'percentage': 0,  # Will update below
                    'first_seen': login_info['timestamp'].isoformat(),
                    'last_seen': login_info['timestamp'].isoformat()
                }
            
            # Update device percentages
            for d_id in profile['devices']:
                profile['devices'][d_id]['percentage'] = (
                    profile['devices'][d_id]['count'] / profile['total_logins'] * 100
                )
            
            # Update location info
            lat_rounded = round(login_info['latitude'], 3)
            lon_rounded = round(login_info['longitude'], 3)
            location_key = f"{lat_rounded}_{lon_rounded}"
            
            if location_key in profile['locations']:
                profile['locations'][location_key]['count'] += 1
                profile['locations'][location_key]['last_seen'] = login_info['timestamp'].isoformat()
            else:
                profile['locations'][location_key] = {
                    'count': 1,
                    'lat': login_info['latitude'],
                    'lon': login_info['longitude'],
                    'first_seen': login_info['timestamp'].isoformat(),
                    'last_seen': login_info['timestamp'].isoformat()
                }
            
            # Update time patterns
            login_hour = str(login_info['timestamp'].hour)
            login_day = str(login_info['timestamp'].weekday())
            
            hour_dist = profile['time_patterns']['hour_distribution']
            hour_dist[login_hour] = hour_dist.get(login_hour, 0) + 1
            
            day_dist = profile['time_patterns']['day_of_week_distribution']
            day_dist[login_day] = day_dist.get(login_day, 0) + 1
        
        # Save updated profile
        self._save_user_profile(user_id, profile)
    
    # Core Analysis Methods
  
    
    def _identify_user_patterns(
        self, 
        user_id: str, 
        user_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Identify patterns in user login behavior using multimodal clustering.
        
        Args:
            user_id: The unique identifier for the user
            user_data: DataFrame containing the user's login data
            
        Returns:
            Dictionary with cluster assignments and statistics
        """
        # Extract features for clustering
        # Location features
        location_data = user_data[['latitude', 'longitude']].copy()
        
        # Handle antimeridian for global banking users
        location_data = handle_antimeridian(location_data)
        
        # Time features - convert to cyclical features to handle time properly
        user_data['hour_sin'] = np.sin(2 * np.pi * pd.to_datetime(user_data['timestamp']).dt.hour / 24)
        user_data['hour_cos'] = np.cos(2 * np.pi * pd.to_datetime(user_data['timestamp']).dt.hour / 24)
        user_data['day_sin'] = np.sin(2 * np.pi * pd.to_datetime(user_data['timestamp']).dt.dayofweek / 7)
        user_data['day_cos'] = np.cos(2 * np.pi * pd.to_datetime(user_data['timestamp']).dt.dayofweek / 7)
        
        # Device ID features - one-hot encode
        device_features = pd.get_dummies(user_data['device_id'], prefix='device')
        
        # Combine all features
        combined_features = pd.concat([
            location_data, 
            user_data[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']], 
            device_features
        ], axis=1)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)
        
        # Try DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        # If DBSCAN identifies only noise, try KMeans
        if len(np.unique(cluster_labels)) <= 1:
            # Try to find optimal K using elbow method
            k_range = range(1, min(10, len(user_data)))
            inertias = []
            
            for k in k_range:
                if k < len(user_data):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(scaled_features)
                    inertias.append(kmeans.inertia_)
            
            # Find elbow point (simplified approach)
            optimal_k = 2  # Default
            if len(inertias) > 2:
                diffs = np.diff(inertias)
                if len(diffs) > 1:
                    optimal_k = np.argmax(np.diff(diffs)) + 2
            
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to original data
        user_data_with_clusters = user_data.copy()
        user_data_with_clusters['cluster'] = cluster_labels
        
        # Compute cluster statistics
        cluster_stats = {}
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
                
            cluster_data = user_data_with_clusters[user_data_with_clusters['cluster'] == cluster_id]
            
            # Calculate geographic mean center for this cluster
            geo_center = calculate_geographic_mean_center(cluster_data[['latitude', 'longitude']])
            
            # Get most common device IDs in this cluster
            devices = cluster_data['device_id'].value_counts().to_dict()
            
            # Time patterns
            time_patterns = {
                'hour_dist': pd.to_datetime(cluster_data['timestamp']).dt.hour.value_counts().sort_index().to_dict(),
                'day_dist': pd.to_datetime(cluster_data['timestamp']).dt.dayofweek.value_counts().sort_index().to_dict()
            }
            
            cluster_stats[str(int(cluster_id))] = {
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(user_data) * 100,
                'geo_center': geo_center,
                'std_distance': calculate_standard_distance(cluster_data[['latitude', 'longitude']]),
                'devices': devices,
                'time_patterns': time_patterns
            }
        
        return {
            'user_data_with_clusters': user_data_with_clusters,
            'cluster_stats': cluster_stats,
            'num_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
            'features': combined_features.columns.tolist()
        }
    
    def _detect_location_anomaly(
        self, 
        login: Dict[str, Any], 
        pattern_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect if a login location is anomalous based on the user's patterns.
        
        Args:
            login: Dictionary with current login information
            pattern_analysis: Dictionary with the user's pattern analysis
            
        Returns:
            Dictionary with anomaly score and details
        """
        # Extract current login location
        current_lat, current_lon = login['latitude'], login['longitude']
        
        # Check if there are any clusters
        if not pattern_analysis.get('cluster_stats', {}):
            return {'score': 0.5, 'details': {'message': 'No clusters available for comparison'}}
        
        # Initialize variables
        min_distance = float('inf')
        closest_cluster = None
        
        # Check distance to each cluster center
        for cluster_id, stats in pattern_analysis['cluster_stats'].items():
            center = stats['geo_center']
            
            # Calculate haversine distance
            distance = haversine_distance(
                center['latitude'],
                center['longitude'],
                current_lat,
                current_lon
            )
            
            # Handle case where distance is a pandas Series
            if isinstance(distance, pd.Series):
                distance = distance.iloc[0]
            
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # Get standard distance for the closest cluster
        std_distance = pattern_analysis['cluster_stats'][closest_cluster]['std_distance']
        
        # Calculate z-score (how many standard deviations away)
        if std_distance > 0:
            z_score = min_distance / std_distance
        else:
            # If std_distance is 0, this is highly anomalous if min_distance > 0
            z_score = 10.0 if min_distance > 0 else 0.0
        
        # Convert to probability score (0 to 1)
        # Using sigmoid function to map z-score to probability
        prob_score = 1 / (1 + np.exp(-0.5 * (z_score - 3)))
        
        return {
            'score': float(prob_score),
            'details': {
                'distance_km': float(min_distance),
                'closest_cluster': closest_cluster,
                'cluster_std_distance': float(std_distance),
                'z_score': float(z_score)
            }
        }
    
    def _detect_device_anomaly(
        self, 
        login: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect if a login device is anomalous based on the user's profile.
        
        Args:
            login: Dictionary with current login information
            user_profile: Dictionary with the user's profile
            
        Returns:
            Dictionary with anomaly score and details
        """
        device_id = login['device_id']
        # Ensure device_id is a hashable type (string)
        if isinstance(device_id, pd.Series):
            device_id = device_id.iloc[0] if not device_id.empty else None
            if device_id is None:
                return {'score': 0.9, 'details': {'device_status': 'error', 'message': 'Invalid device ID'}}
            device_id = str(device_id)
        
        # Check if devices field is a string, parse it first if needed
        if isinstance(user_profile['devices'], str):
            try:
                user_profile['devices'] = json.loads(user_profile['devices'])
            except json.JSONDecodeError:
                return {'score': 0.9, 'details': {'device_status': 'error', 'message': 'Invalid devices format'}}
        
        # Check if device has been seen before
        if device_id in user_profile['devices']:
            # Get device info and ensure it's a dictionary
            device_info = user_profile['devices'][device_id]
            
            # Parse JSON string if necessary
            if isinstance(device_info, str):
                try:
                    device_info = json.loads(device_info)
                    user_profile['devices'][device_id] = device_info  # Update with parsed value
                except json.JSONDecodeError:
                    # Handle the case where the string is not valid JSON
                    return {'score': 0.9, 'details': {'device_status': 'error', 'message': 'Invalid device info format'}}
            
            # Known device - calculate score based on frequency
            frequency = device_info['percentage'] / 100
            # Inverse relationship - less frequently used devices are more suspicious
            device_score = 1 - frequency
            
            details = {
                'device_status': 'known',
                'previous_usage_count': device_info['count'],
                'previous_usage_percent': device_info['percentage']
            }
        else:
            # New device is highly suspicious
            device_score = 0.9
            details = {
                'device_status': 'new',
                'previous_usage_count': 0,
                'previous_usage_percent': 0
            }
        
        return {
            'score': float(device_score),
            'details': details
        }
    
    def _detect_time_anomaly(
        self, 
        login: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect if a login time is anomalous based on the user's profile.
        
        Args:
            login: Dictionary with current login information
            user_profile: Dictionary with the user's profile
            
        Returns:
            Dictionary with anomaly score and details
        """
        login_time = pd.to_datetime(login['timestamp'])
        if isinstance(login_time, pd.Series):
            hour = login_time.dt.hour.iloc[0]
            day_of_week = login_time.dt.dayofweek.iloc[0]
        else:
            hour = login_time.hour
            day_of_week = login_time.weekday()
        
        # Check if time_patterns is a string and parse it if needed
        if isinstance(user_profile['time_patterns'], str):
            try:
                user_profile['time_patterns'] = json.loads(user_profile['time_patterns'])
            except json.JSONDecodeError:
                return {'score': 0.5, 'details': {'message': 'Invalid time patterns format'}}
        
        # Check hour distribution
        hour_dist = user_profile['time_patterns']['hour_distribution']
        day_dist = user_profile['time_patterns']['day_of_week_distribution']
        
        # Calculate hour anomaly
        total_logins = user_profile['total_logins']
        hour_count = hour_dist.get(str(hour), 0)
        hour_freq = hour_count / total_logins if total_logins > 0 else 0
        
        # Calculate day anomaly
        day_count = day_dist.get(str(day_of_week), 0)
        day_freq = day_count / total_logins if total_logins > 0 else 0
        
        # Combine scores (weighted more heavily toward hour)
        combined_freq = 0.7 * hour_freq + 0.3 * day_freq
        
        # Convert to anomaly score (0 to 1)
        time_score = 1 - combined_freq
        
        return {
            'score': float(time_score),
            'details': {
                'login_hour': int(hour),
                'login_day': int(day_of_week),
                'hour_frequency': float(hour_freq),
                'day_frequency': float(day_freq),
                'hour_count': int(hour_count),
                'day_count': int(day_count)
            }
        }
    
    def _detect_velocity_anomaly(
        self, 
        login: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect impossible travel scenarios based on login velocity.
        
        Args:
            login: Dictionary with current login information
            user_profile: Dictionary with the user's profile
            
        Returns:
            Dictionary with anomaly score and details
        """
        # Need at least the current login and one previous login
        if 'last_login' not in user_profile or user_profile['total_logins'] < 1:
            return {'score': 0.0, 'details': {'message': 'Insufficient login history'}}
        
        # Calculate time difference between current login and last known login
        current_time = pd.to_datetime(login['timestamp'])
        last_login_time = pd.to_datetime(user_profile['last_login'])
        
        # Handle the case where current_time or last_login_time is a Series
        if isinstance(current_time, pd.Series):
            current_time = current_time.iloc[0]
        if isinstance(last_login_time, pd.Series):
            last_login_time = last_login_time.iloc[0]
            
        # Make both datetime objects timezone-naive to avoid timezone comparison errors
        if current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
        if last_login_time.tzinfo is not None:
            last_login_time = last_login_time.replace(tzinfo=None)
        
        time_diff_hours = (current_time - last_login_time).total_seconds() / 3600
        
        # Check if locations field is a string, parse it first if needed
        if isinstance(user_profile['locations'], str):
            try:
                user_profile['locations'] = json.loads(user_profile['locations'])
            except json.JSONDecodeError:
                return {'score': 0.0, 'details': {'message': 'Invalid locations format'}}
        
        # Find the last known location
        last_known_location = None
        for loc_key, loc_data in user_profile['locations'].items():
            if pd.to_datetime(loc_data['last_seen']) == last_login_time:
                last_known_location = {'latitude': loc_data['lat'], 'longitude': loc_data['lon']}
                break
        
        if last_known_location is None:
            # If we can't find the exact last location, use any recent location
            recent_locations = sorted(
                user_profile['locations'].items(),
                key=lambda x: abs((pd.to_datetime(x[1]['last_seen']).replace(tzinfo=None) - last_login_time.replace(tzinfo=None)).total_seconds())
            )
            
            if recent_locations:
                loc_data = recent_locations[0][1]
                last_known_location = {'latitude': loc_data['lat'], 'longitude': loc_data['lon']}
            else:
                return {'score': 0.0, 'details': {'message': 'Last location unknown'}}
        
        # Calculate distance between last location and current location
        distance_km = haversine_distance(
            last_known_location['latitude'],
            last_known_location['longitude'],
            login['latitude'],
            login['longitude']
        )
        
        # Calculate required velocity (km/h)
        if time_diff_hours > 0:
            velocity = distance_km / time_diff_hours
        else:
            return {'score': 1.0, 'details': {'message': 'Zero or negative time difference - highly suspicious'}}
        
        # Define thresholds for suspicious velocity
        # Commercial flights ~800 km/h, high-speed trains ~300 km/h
        if velocity > 1000:  # Impossible travel
            velocity_score = 1.0
        elif velocity > 800:  # Faster than commercial flight
            velocity_score = 0.9
        elif velocity > 300:  # Commercial flight
            velocity_score = 0.6
        elif velocity > 150:  # High-speed train
            velocity_score = 0.3
        elif velocity > 80:   # Car on highway
            velocity_score = 0.1
        else:  # Normal travel
            velocity_score = 0.0
        
        return {
            'score': float(velocity_score),
            'details': {
                'distance_km': float(distance_km),
                'time_diff_hours': float(time_diff_hours),
                'velocity_km_h': float(velocity),
                'prev_lat': float(last_known_location['latitude']),
                'prev_lon': float(last_known_location['longitude'])
            }
        }
    
    def _detect_anomalies_with_isolation_forest(
        self, 
        login: Dict[str, Any], 
        user_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Use Isolation Forest to detect anomalies.
        
        Args:
            login: Dictionary with current login information
            user_data: DataFrame with the user's historical data
            
        Returns:
            Dictionary with anomaly score and details
        """
        # Minimum data points needed
        if len(user_data) < 10:
            return {'score': 0.5, 'details': {'message': 'Insufficient data for Isolation Forest'}}
        
        try:
            # Prepare features for historical data
            features = user_data[['latitude', 'longitude']].copy()
            
            # Add time features
            user_data['hour'] = pd.to_datetime(user_data['timestamp']).dt.hour
            user_data['day_of_week'] = pd.to_datetime(user_data['timestamp']).dt.dayofweek
            
            features['hour_sin'] = np.sin(2 * np.pi * user_data['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * user_data['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * user_data['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * user_data['day_of_week'] / 7)
            
            # One-hot encode device ID
            device_features = pd.get_dummies(user_data['device_id'], prefix='device')
            features = pd.concat([features, device_features], axis=1)
            
            # Prepare current login features
            current_hour = pd.to_datetime(login['timestamp']).hour
            current_day = pd.to_datetime(login['timestamp']).dayofweek
            
            current_features = pd.DataFrame({
                'latitude': [login['latitude']],
                'longitude': [login['longitude']],
                'hour_sin': [np.sin(2 * np.pi * current_hour / 24)],
                'hour_cos': [np.cos(2 * np.pi * current_hour / 24)],
                'day_sin': [np.sin(2 * np.pi * current_day / 7)],
                'day_cos': [np.cos(2 * np.pi * current_day / 7)]
            })
            
            # Add device features to current login
            device_id = login['device_id']
            for col in device_features.columns:
                device_prefix = f"device_{device_id}"
                if col == device_prefix:
                    current_features[col] = 1
                else:
                    current_features[col] = 0
            
            # Ensure current_features has all columns from features
            for col in features.columns:
                if col not in current_features.columns:
                    current_features[col] = 0
            
            # Ensure columns are in the same order
            current_features = current_features[features.columns]
            
            # Train Isolation Forest
            isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
            isolation_forest.fit(features)
            
            # Get anomaly score for current login
            anomaly_score = isolation_forest.decision_function(current_features)
            
            # Isolation Forest returns negative scores for anomalies, positive for normal points
            # Convert to 0-1 range where 1 is anomalous
            normalized_score = 1 - (anomaly_score[0] + 0.5) / 1.0
            normalized_score = max(0, min(1, normalized_score))
            
            return {
                'score': float(normalized_score),
                'details': {
                    'raw_score': float(anomaly_score[0])
                }
            }
        except Exception as e:
            logger.error(f"Error in Isolation Forest: {str(e)}")
            return {'score': 0.5, 'details': {'message': f'Error in Isolation Forest: {str(e)}'}}
    
    def _calculate_mahalanobis_anomaly(
        self, 
        login: Dict[str, Any], 
        user_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate Mahalanobis distance for multivariate outlier detection.
        
        Args:
            login: Dictionary with current login information
            user_data: DataFrame with the user's historical data
            
        Returns:
            Dictionary with anomaly score and details
        """
        if len(user_data) < 5:
            return {'score': 0.5, 'details': {'message': 'Insufficient data for Mahalanobis distance'}}
        
        try:
            # Select numerical features
            features = user_data[['latitude', 'longitude']].copy()
            
            # Add time features
            user_data['hour'] = pd.to_datetime(user_data['timestamp']).dt.hour
            features['hour'] = user_data['hour']
            
            # Calculate mean vector and covariance matrix
            mean_vector = features.mean().values
            cov_matrix = features.cov().values
            
            # Handle singular covariance matrix
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Add small regularization term
                cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Current login features
            current_vector = np.array([
                login['latitude'], 
                login['longitude'], 
                pd.to_datetime(login['timestamp']).hour
            ])
            
            # Calculate Mahalanobis distance
            mahalanobis_dist = mahalanobis(current_vector, mean_vector, inv_cov_matrix)
            
            # Convert to probability using chi-squared distribution
            # Degrees of freedom = number of features
            p_value = 1 - stats.chi2.cdf(mahalanobis_dist, len(features.columns))
            
            # Convert p-value to anomaly score (0 to 1, where 1 is anomalous)
            anomaly_score = 1 - p_value
            
            return {
                'score': float(anomaly_score),
                'details': {
                    'mahalanobis_distance': float(mahalanobis_dist),
                    'p_value': float(p_value),
                    'degrees_of_freedom': len(features.columns)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating Mahalanobis distance: {str(e)}")
            return {'score': 0.5, 'details': {'message': f'Error calculating Mahalanobis distance: {str(e)}'}}
    

    # Utility Methods
    
    
    def _calculate_combined_variance(self, login_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate combined variance metrics for user locations.
        
        Args:
            login_data: DataFrame with user login data
            
        Returns:
            Dictionary with various variance metrics
        """
        if len(login_data) <= 1:
            return {
                'standard_distance': 0.0,
                'robust_distance': 0.0,
                'weighted_distance': 0.0,
                'multimodal_distance': 0.0,
                'combined_score': 0.0
            }
        
        # Basic standard distance
        std_distance = calculate_standard_distance(login_data[['latitude', 'longitude']])
        
        # For simplicity, we'll use the standard distance for all metrics in this implementation
        # In a full implementation, you would calculate different types of variance
        
        combined_score = std_distance
        
        return {
            'standard_distance': std_distance,
            'robust_distance': std_distance,  # Simplified
            'weighted_distance': std_distance,  # Simplified
            'multimodal_distance': std_distance,  # Simplified
            'combined_score': combined_score
        }
    
    def _analyze_login_velocity(self, login_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the velocity between consecutive logins.
        
        Args:
            login_data: DataFrame containing login history
            
        Returns:
            Dictionary with login velocity statistics
        """
        # Sort by timestamp
        login_data = login_data.sort_values('timestamp')
        
        if len(login_data) <= 1:
            return {
                'mean_time_between_logins': None,
                'std_time_between_logins': None,
                'min_time_between_logins': None,
                'max_time_between_logins': None,
                'median_time_between_logins': None
            }
        
        # Calculate time differences between consecutive logins
        login_data = login_data.copy()
        login_data['next_timestamp'] = login_data['timestamp'].shift(-1)
        login_data['time_diff'] = (login_data['next_timestamp'] - 
                                  login_data['timestamp']).dt.total_seconds() / 3600  # in hours
        
        # Remove the last row (no next timestamp)
        time_diffs = login_data['time_diff'].dropna()
        
        return {
            'mean_time_between_logins': float(np.mean(time_diffs)),
            'std_time_between_logins': float(np.std(time_diffs)),
            'min_time_between_logins': float(np.min(time_diffs)),
            'max_time_between_logins': float(np.max(time_diffs)),
            'median_time_between_logins': float(np.median(time_diffs))
        }