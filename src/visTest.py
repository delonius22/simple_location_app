# visualization_with_trajectories.py

import folium
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson
import pandas as pd
import numpy as np
import os
import logging
import traceback
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_with_trajectories(data_df, save_path="user_locations_with_trajectories.html"):
    """Create a visualization with user search functionality and trajectory lines."""
    logger.info("Starting visualization process with search capability and trajectories")
    
    try:
        # Initial data validation
        if not isinstance(data_df, pd.DataFrame) or len(data_df) == 0:
            logger.error("Invalid or empty data")
            return None
            
        # Ensure required columns exist
        required_columns = ['user_id', 'latitude', 'longitude', 'timestamp', 'device_id']
        for col in required_columns:
            if col not in data_df.columns:
                logger.error(f"Required column '{col}' missing from input data")
                return None
        
        # Ensure timestamp is datetime
        try:
            if data_df['timestamp'].dtype != 'datetime64[ns]':
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                logger.info("Converted timestamp column to datetime")
        except Exception as e:
            logger.error(f"Error converting timestamps: {e}")
            data_df['timestamp'] = pd.Timestamp.now()
        
        # Add missing columns with default values if needed
        if 'is_anomalous' not in data_df.columns:
            data_df['is_anomalous'] = False
        
        if 'anomaly_score' not in data_df.columns:
            data_df['anomaly_score'] = 0.0
        
        # Clean coordinates
        try:
            # Convert to numeric, coerce errors to NaN
            data_df['latitude'] = pd.to_numeric(data_df['latitude'], errors='coerce')
            data_df['longitude'] = pd.to_numeric(data_df['longitude'], errors='coerce')
            
            # Drop rows with NaN coordinates
            invalid_rows = data_df[data_df['latitude'].isna() | data_df['longitude'].isna()].shape[0]
            if invalid_rows > 0:
                logger.warning(f"Dropping {invalid_rows} rows with invalid coordinates")
                data_df = data_df.dropna(subset=['latitude', 'longitude'])
            
            # Validate coordinate ranges
            data_df = data_df[(data_df['latitude'] >= -90) & (data_df['latitude'] <= 90) & 
                              (data_df['longitude'] >= -180) & (data_df['longitude'] <= 180)]
            logger.info(f"Valid coordinates for {len(data_df)} rows")
            
            if len(data_df) == 0:
                logger.error("No rows with valid coordinates remain")
                return None
        except Exception as e:
            logger.error(f"Error processing coordinates: {e}")
            return None
        
        # Calculate map center
        try:
            center_lat = float(data_df['latitude'].mean())
            center_lon = float(data_df['longitude'].mean())
        except Exception as e:
            logger.error(f"Error calculating map center: {e}")
            center_lat, center_lon = 0, 0  # Default to null island
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], 
                     zoom_start=10, 
                     tiles='CartoDB positron')
        
        # Get unique user IDs
        unique_users = sorted(data_df['user_id'].unique())
        
        # Create feature groups for each user ID
        user_groups = {}
        for user_id in unique_users:
            user_groups[user_id] = folium.FeatureGroup(name=f"User: {user_id}", show=False)
            m.add_child(user_groups[user_id])
        
        # Create general feature groups
        all_markers = folium.FeatureGroup(name="All Logins", show=True)
        anomaly_markers = folium.FeatureGroup(name="Anomalies Only", show=False)
        heatmap_layer = folium.FeatureGroup(name="Heatmap", show=True)
        
        # Add markers for each data point
        normal_count = 0
        anomaly_count = 0
        
        for idx, row in data_df.iterrows():
            try:
                # Get coordinates
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                
                # Get user ID
                user_id = str(row['user_id'])
                
                # Format timestamp
                if isinstance(row['timestamp'], pd.Timestamp):
                    time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(row['timestamp'])
                
                # Create popup content
                popup_html = f"""
                <div style="font-family: sans-serif; width: 200px;">
                    <h4 style="margin-bottom: 10px;">User Login Details</h4>
                    <b>User ID:</b> {user_id}<br>
                    <b>Device:</b> {row['device_id']}<br>
                    <b>Time:</b> {time_str}<br>
                    <b>Coordinates:</b> [{lat:.5f}, {lon:.5f}]<br>
                    <b>Anomaly:</b> {'Yes' if row.get('is_anomalous', False) else 'No'}<br>
                    <b>Score:</b> {float(row.get('anomaly_score', 0.0)):.4f}
                </div>
                """
                
                # Create marker with appropriate styling
                is_anomaly = bool(row.get('is_anomalous', False))
                
                if is_anomaly:
                    icon = folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
                    anomaly_count += 1
                else:
                    icon = folium.Icon(color='blue', icon='user', prefix='fa')
                    normal_count += 1
                
                marker = folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=icon,
                    tooltip=f"{'Anomaly' if is_anomaly else 'Normal'}: User {user_id}"
                )
                
                # Add to appropriate groups
                marker.add_to(user_groups[user_id])  # User-specific group
                marker.add_to(all_markers)  # All markers group
                
                if is_anomaly:
                    marker.add_to(anomaly_markers)  # Anomaly group
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        # Add heatmap
        try:
            heatmap_data = []
            for _, row in data_df.iterrows():
                try:
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    score = float(row.get('anomaly_score', 0.0))
                    
                    # Scale the weight by anomaly score
                    weight = score * 10
                    heatmap_data.append([lat, lon, weight])
                except:
                    continue
            
            HeatMap(
                data=heatmap_data,
                radius=15,
                gradient={0.1: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1: 'red'},
                min_opacity=0.5,
                max_zoom=1
            ).add_to(heatmap_layer)
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
        
        # Add general feature groups to map
        m.add_child(all_markers)
        m.add_child(anomaly_markers)
        m.add_child(heatmap_layer)
        
        # Create TimestampedGeoJson features for each user
        timestamped_features = {}
        
        for user_id in unique_users:
            # Filter data for this user
            user_data = data_df[data_df['user_id'] == user_id].sort_values('timestamp')
            
            # Skip if not enough data points
            if len(user_data) < 2:
                continue
                
            features = []
            
            # Create point features
            for idx, row in user_data.iterrows():
                try:
                    # Skip invalid coordinates
                    if (pd.isna(row['latitude']) or pd.isna(row['longitude']) or 
                        not (-90 <= float(row['latitude']) <= 90) or 
                        not (-180 <= float(row['longitude']) <= 180)):
                        continue
                    
                    # Extract properties
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    device_id = str(row['device_id'])
                    is_anomaly = bool(row.get('is_anomalous', False))
                    anomaly_score = float(row.get('anomaly_score', 0.0))
                    
                    # Format timestamp
                    if isinstance(row['timestamp'], pd.Timestamp):
                        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        time_str = str(row['timestamp'])
                    
                    # Create point feature
                    point_feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [lon, lat]  # GeoJSON uses [lon, lat] order
                        },
                        'properties': {
                            'time': time_str,
                            'icon': 'circle',
                            'iconstyle': {
                                'fillColor': 'red' if is_anomaly else 'blue',
                                'fillOpacity': 0.8,
                                'stroke': 'true',
                                'radius': 8 if is_anomaly else 5
                            },
                            'style': {'weight': 0},
                            'popup': f"User: {user_id}<br>Device: {device_id}<br>Time: {time_str}<br>Anomaly: {'Yes' if is_anomaly else 'No'}<br>Score: {anomaly_score:.4f}"
                        }
                    }
                    
                    features.append(point_feature)
                    
                except Exception as e:
                    logger.warning(f"Error processing time feature for row {idx}: {e}")
                    continue
            
            # Create line features (trajectories) connecting consecutive points
            if len(features) >= 2:
                for i in range(len(features) - 1):
                    try:
                        point1 = features[i]['geometry']['coordinates']
                        point2 = features[i+1]['geometry']['coordinates']
                        time1 = features[i]['properties']['time']
                        time2 = features[i+1]['properties']['time']
                        
                        # Create line feature
                        line_feature = {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': [point1, point2]
                            },
                            'properties': {
                                'time': time1,  # Use time of first point
                                'style': {
                                    'color': '#3388ff',
                                    'weight': 3,
                                    'opacity': 0.8
                                },
                                'popup': f"User: {user_id}<br>From: {time1}<br>To: {time2}"
                            }
                        }
                        
                        features.append(line_feature)
                        
                    except Exception as e:
                        logger.warning(f"Error creating line feature: {e}")
                        continue
            
            # Store features for this user
            if features:
                timestamped_features[user_id] = features
        
        # Add TimestampedGeoJson for all users combined (initially hidden)
        try:
            # Combine all features
            all_features = []
            for user_features in timestamped_features.values():
                all_features.extend(user_features)
            
            if all_features:
                # Determine time range
                all_timestamps = [
                    datetime.strptime(feature['properties']['time'], '%Y-%m-%d %H:%M:%S')
                    for feature in all_features
                    if 'time' in feature['properties']
                ]
                
                start_time = min(all_timestamps)
                end_time = max(all_timestamps)
                time_range = end_time - start_time
                
                # Set appropriate period and duration
                if time_range.days > 30:
                    period = 'P1D'  # 1 day
                    duration = 'P5D'  # 5 days
                elif time_range.days > 7:
                    period = 'PT12H'  # 12 hours
                    duration = 'P1D'  # 1 day
                elif time_range.days > 1:
                    period = 'PT3H'  # 3 hours
                    duration = 'PT12H'  # 12 hours
                else:
                    period = 'PT30M'  # 30 minutes
                    duration = 'PT3H'  # 3 hours
                
                # Create TimestampedGeoJson for all users
                all_time_slider = TimestampedGeoJson(
                    {
                        'type': 'FeatureCollection',
                        'features': all_features
                    },
                    period=period,
                    duration=duration,
                    auto_play=False,
                    loop=False,
                    max_speed=10,
                    loop_button=True,
                    date_options='YYYY-MM-DD HH:mm:ss',
                    time_slider_drag_update=True,
                    add_last_point=True
                )
                
                # Create a feature group for the time slider
                time_slider_group = folium.FeatureGroup(name="Time Animation (All Users)", show=False)
                all_time_slider.add_to(time_slider_group)
                time_slider_group.add_to(m)
                
                logger.info(f"Added TimestampedGeoJson with {len(all_features)} features")
                
                # Create individual time sliders for each user
                for user_id, features in timestamped_features.items():
                    if features:
                        user_time_slider = TimestampedGeoJson(
                            {
                                'type': 'FeatureCollection',
                                'features': features
                            },
                            period=period,
                            duration=duration,
                            auto_play=False,
                            loop=False,
                            max_speed=10,
                            loop_button=True,
                            date_options='YYYY-MM-DD HH:mm:ss',
                            time_slider_drag_update=True,
                            add_last_point=True
                        )
                        
                        # Create a feature group for this user's time slider
                        user_time_group = folium.FeatureGroup(name=f"Time Animation: {user_id}", show=False)
                        user_time_slider.add_to(user_time_group)
                        user_time_group.add_to(m)
                        
                        logger.info(f"Added TimestampedGeoJson for user {user_id} with {len(features)} features")
            else:
                logger.warning("No valid features for TimestampedGeoJson")
                
        except Exception as e:
            logger.error(f"Error adding TimestampedGeoJson: {e}")
            logger.error(traceback.format_exc())
            # Continue without time slider
        
        # Create search interface
        search_html = f"""
        <div id="search-container" style="position: fixed; 
                                        top: 10px; 
                                        left: 60px; 
                                        z-index: 1000; 
                                        background-color: white; 
                                        padding: 10px; 
                                        border-radius: 4px; 
                                        box-shadow: 0 1px 5px rgba(0,0,0,0.4);">
            <div style="margin-bottom: 8px; font-weight: bold; color: #444;">
                Search by User ID
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <input type="text" id="user-search" placeholder="Enter user ID..." 
                    style="padding: 6px; 
                          border: 1px solid #ccc; 
                          border-radius: 4px; 
                          margin-right: 5px; 
                          width: 150px;"/>
                <button onclick="searchUser()" 
                    style="padding: 6px 10px; 
                          background-color: #4CAF50; 
                          color: white; 
                          border: none; 
                          border-radius: 4px; 
                          cursor: pointer;">
                    Search
                </button>
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-size: 13px;">Available Users:</span>
                <select id="user-dropdown" onchange="selectUser(this.value)" 
                    style="padding: 6px; 
                          border: 1px solid #ccc; 
                          border-radius: 4px; 
                          width: 100%; 
                          margin-top: 4px;">
                    <option value="">-- Select User --</option>
                    {' '.join([f'<option value="{uid}">{uid}</option>' for uid in unique_users])}
                </select>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <button onclick="showTrajectory()" 
                    style="padding: 6px 10px; 
                          background-color: #2196F3; 
                          color: white; 
                          border: none; 
                          border-radius: 4px; 
                          cursor: pointer;
                          flex: 1;
                          margin-right: 5px;">
                    Show Trajectory
                </button>
                <button onclick="resetView()" 
                    style="padding: 6px 10px; 
                          background-color: #f44336; 
                          color: white; 
                          border: none; 
                          border-radius: 4px; 
                          cursor: pointer;
                          flex: 1;">
                    Reset View
                </button>
            </div>
            <div style="margin-top: 8px; font-size: 12px; color: #666;">
                <div><b>Total Points:</b> {len(data_df)}</div>
                <div><b>Normal:</b> {normal_count}</div>
                <div><b>Anomalies:</b> {anomaly_count}</div>
                <div><b>Time Period:</b> {data_df['timestamp'].min().strftime('%Y-%m-%d')} to {data_df['timestamp'].max().strftime('%Y-%m-%d')}</div>
            </div>
        </div>
        """
        
        # Create JavaScript for search and trajectory functionality
        search_js = """
        <script>
        var currentUserId = null;
        
        function searchUser() {
            var userId = document.getElementById('user-search').value.trim();
            if (!userId) {
                alert("Please enter a user ID");
                return;
            }
            
            currentUserId = userId;
            
            // Find the user layer in the layer control
            var found = false;
            var layerControl = document.getElementsByClassName('leaflet-control-layers-overlays')[0];
            var inputs = layerControl.getElementsByTagName('input');
            var labels = layerControl.getElementsByTagName('span');
            
            // Hide all layers first
            for (var i = 0; i < inputs.length; i++) {
                var labelText = labels[i].textContent.trim();
                inputs[i].checked = false;
                
                // Show only the searched user's layer
                if (labelText === 'User: ' + userId) {
                    inputs[i].checked = true;
                    found = true;
                }
                
                // Trigger change event
                var event = new Event('change');
                inputs[i].dispatchEvent(event);
            }
            
            if (!found) {
                alert("User ID not found: " + userId);
                resetView();
            } else {
                // Update dropdown to match
                document.getElementById('user-dropdown').value = userId;
            }
        }
        
        function showTrajectory() {
            if (!currentUserId) {
                alert("Please select a user first");
                return;
            }
            
            var userId = currentUserId;
            var layerControl = document.getElementsByClassName('leaflet-control-layers-overlays')[0];
            var inputs = layerControl.getElementsByTagName('input');
            var labels = layerControl.getElementsByTagName('span');
            
            // Show only the selected user's time animation layer
            for (var i = 0; i < inputs.length; i++) {
                var labelText = labels[i].textContent.trim();
                
                // Hide all layers
                inputs[i].checked = false;
                
                // Show only the time animation for the selected user
                if (labelText === 'Time Animation: ' + userId) {
                    inputs[i].checked = true;
                    
                    // Trigger the change event
                    var event = new Event('change');
                    inputs[i].dispatchEvent(event);
                    
                    // Find and click the play button (after a short delay to ensure layer is visible)
                    setTimeout(function() {
                        var playButton = document.querySelector('.timecontrol-play');
                        if (playButton) {
                            playButton.click();
                        }
                    }, 300);
                }
            }
        }
        
        function selectUser(userId) {
            if (userId) {
                document.getElementById('user-search').value = userId;
                searchUser();
            } else {
                resetView();
            }
        }
        
        function resetView() {
            // Clear the search field and dropdown
            document.getElementById('user-search').value = "";
            document.getElementById('user-dropdown').value = "";
            currentUserId = null;
            
            // Show only the default layers
            var layerControl = document.getElementsByClassName('leaflet-control-layers-overlays')[0];
            var inputs = layerControl.getElementsByTagName('input');
            var labels = layerControl.getElementsByTagName('span');
            
            for (var i = 0; i < inputs.length; i++) {
                var labelText = labels[i].textContent.trim();
                
                // Reset all layer visibility
                if (labelText === 'All Logins' || labelText === 'Heatmap') {
                    inputs[i].checked = true;
                } else {
                    inputs[i].checked = false;
                }
                
                // Trigger change event
                var event = new Event('change');
                inputs[i].dispatchEvent(event);
            }
        }
        
        // Initialize with default view
        window.onload = function() {
            // Short delay to ensure map is fully loaded
            setTimeout(resetView, 500);
        };
        </script>
        """
        
        # Add search interface and functionality to map
        m.get_root().html.add_child(folium.Element(search_html))
        m.get_root().html.add_child(folium.Element(search_js))
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                   bottom: 50px; right: 50px; 
                   width: 200px; 
                   border:2px solid grey; z-index:9999; 
                   font-size:14px;
                   background-color: white; 
                   padding: 10px;
                   border-radius: 5px;">
            <p style="margin-top: 0;"><b>Legend</b></p>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <i class="fa fa-user fa-1x" style="color:blue; margin-right: 8px;"></i>
                <span>Normal Login</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <i class="fa fa-exclamation-circle fa-1x" style="color:red; margin-right: 8px;"></i>
                <span>Anomalous Login</span>
            </div>
            <div style="margin-top: 5px; display: flex; align-items: center; margin-bottom: 5px;">
                <div style="height: 10px; width: 50px; background: linear-gradient(to right, blue, lime, yellow, orange, red); display: inline-block; margin-right: 8px;"></div>
                <span>Anomaly Score</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 3px; background-color: #3388ff; margin-right: 8px;"></div>
                <span>User Trajectory</span>
            </div>
            <div style="font-size: 12px; margin-top: 5px; border-top: 1px solid #ccc; padding-top: 5px;">
                <div>Use "Show Trajectory" to view a user's movement pattern over time</div>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            m.save(save_path)
            logger.info(f"Map saved to {save_path}")
            
            if os.path.exists(save_path):
                return m
            else:
                logger.error(f"Failed to save map to {save_path}")
                return None
        except Exception as e:
            logger.error(f"Error saving map: {e}")
            logger.error(traceback.format_exc())
            
            # Try alternate location
            try:
                alt_path = os.path.join(os.path.expanduser("~"), "user_trajectories_map.html")
                m.save(alt_path)
                logger.info(f"Map saved to alternate location: {alt_path}")
                return m
            except:
                return m  # Return unsaved map as last resort
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    n = 100
    
    # Generate test data for each user with realistic movement patterns
    users = ['user1', 'user2', 'user3']
    all_data = []
    
    for user_id in users:
        # Create a base location for each user
        if user_id == 'user1':
            base_lat, base_lon = 37.7749, -122.4194  # San Francisco
        elif user_id == 'user2':
            base_lat, base_lon = 40.7128, -74.0060  # New York
        else:
            base_lat, base_lon = 34.0522, -118.2437  # Los Angeles
            
        # Create sequential timestamps for this user
        timestamps = pd.date_range(start='2023-01-01', periods=n//len(users), freq='3H')
        
        # Create trajectory - simulate a user moving between several locations
        lats = []
        lons = []
        
        # Create 3-5 common locations for this user
        num_locations = np.random.randint(3, 6)
        common_locs = []
        
        for _ in range(num_locations):
            # Each location is within ~5km of the base
            loc_lat = base_lat + np.random.normal(0, 0.05)
            loc_lon = base_lon + np.random.normal(0, 0.05)
            common_locs.append((loc_lat, loc_lon))
        
        # Assign each timestamp to one of the common locations
        # with occasional random locations (potentially anomalous)
        for i in range(len(timestamps)):
            if np.random.random() < 0.1:  # 10% chance of a random location
                lat = base_lat + np.random.normal(0, 0.2)  # Wider distribution
                lon = base_lon + np.random.normal(0, 0.2)
                is_anomaly = True
            else:
                # Pick a common location
                loc_idx = np.random.randint(0, len(common_locs))
                loc_lat, loc_lon = common_locs[loc_idx]
                
                # Add small random variation
                lat = loc_lat + np.random.normal(0, 0.005)
                lon = loc_lon + np.random.normal(0, 0.005)
                is_anomaly = False
            
            lats.append(lat)
            lons.append(lon)
            
            # Create a record
            device_id = np.random.choice(['desktop', 'mobile', 'tablet'])
            anomaly_score = np.random.uniform(0.7, 1.0) if is_anomaly else np.random.uniform(0.0, 0.3)
            
            all_data.append({
                'user_id': user_id,
                'timestamp': timestamps[i],
                'latitude': lat,
                'longitude': lon,
                'device_id': device_id,
                'is_anomalous': is_anomaly,
                'anomaly_score': anomaly_score
            })
    
    # Convert to DataFrame
    data = pd.DataFrame(all_data)
    
    # Create test visualization
    result = visualize_with_trajectories(data)
    
    if result:
        print("Test completed successfully")
    else:
        print("Test failed")