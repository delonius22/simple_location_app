# visualization_with_folium_search.py

import folium
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson, Search
import pandas as pd
import numpy as np
import os
import logging
import traceback
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_with_folium_search(data_df, save_path="user_locations_search.html"):
    """Create a visualization with Folium's built-in search functionality."""
    logger.info("Starting visualization process with Folium search plugin")
    
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
        
        # Create feature groups for different views
        all_markers_fg = folium.FeatureGroup(name="All Logins")
        anomaly_fg = folium.FeatureGroup(name="Anomalies Only")
        heatmap_fg = folium.FeatureGroup(name="Heatmap")
        
        # Dictionary to store user-specific feature groups
        user_fgs = {}
        
        # GeoJSON data for search plugin
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Process data points
        for idx, row in data_df.iterrows():
            try:
                # Validate coordinates
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                
                # Skip invalid coordinates
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                
                # Get user ID and format timestamp
                user_id = str(row['user_id'])
                device_id = str(row['device_id'])
                is_anomaly = bool(row.get('is_anomalous', False))
                
                if isinstance(row['timestamp'], pd.Timestamp):
                    time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(row['timestamp'])
                
                # Create popup content
                popup_content = f"""
                <div style="font-family: sans-serif; width: 200px;">
                    <h4 style="margin-bottom: 10px;">User Login Details</h4>
                    <b>User ID:</b> {user_id}<br>
                    <b>Device ID:</b> {device_id}<br>
                    <b>Timestamp:</b> {time_str}<br>
                    <b>Coordinates:</b> [{lat:.5f}, {lon:.5f}]<br>
                    <b>Anomaly:</b> {'Yes' if is_anomaly else 'No'}<br>
                    <b>Anomaly Score:</b> {float(row.get('anomaly_score', 0.0)):.4f}
                </div>
                """
                
                # Create marker with appropriate icon
                if is_anomaly:
                    icon = folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
                else:
                    icon = folium.Icon(color='blue', icon='user', prefix='fa')
                
                marker = folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=icon,
                    tooltip=f"{'Anomaly' if is_anomaly else 'Normal'}: User {user_id}"
                )
                
                # Add to all markers group
                marker.add_to(all_markers_fg)
                
                # Add to anomaly group if applicable
                if is_anomaly:
                    marker.add_to(anomaly_fg)
                
                # Get or create user-specific feature group
                if user_id not in user_fgs:
                    user_fgs[user_id] = folium.FeatureGroup(name=f"User: {user_id}")
                
                # Add to user-specific group
                marker.add_to(user_fgs[user_id])
                
                # Create GeoJSON feature for this point (for search functionality)
                geojson_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "user_id": user_id,
                        "device_id": device_id,
                        "timestamp": time_str,
                        "is_anomalous": "Yes" if is_anomaly else "No",
                        "anomaly_score": float(row.get('anomaly_score', 0.0)),
                        "name": f"User: {user_id}"  # Important for search plugin
                    }
                }
                
                geojson_data["features"].append(geojson_feature)
                
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
                    
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        continue
                        
                    score = float(row.get('anomaly_score', 0.0))
                    weight = score * 10  # Scale for visibility
                    
                    heatmap_data.append([lat, lon, weight])
                except:
                    continue
            
            # Add heatmap to its feature group
            if heatmap_data:
                HeatMap(
                    data=heatmap_data,
                    radius=15,
                    gradient={0.1: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1: 'red'},
                    min_opacity=0.5,
                    max_zoom=1
                ).add_to(heatmap_fg)
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
        
        # Add time-based visualization with trajectory lines
        try:
            # Process each user for time-based visualization
            for user_id, user_fg in user_fgs.items():
                # Filter data for this user
                user_data = data_df[data_df['user_id'] == user_id].sort_values('timestamp')
                
                # Skip if too few points
                if len(user_data) < 2:
                    continue
                
                # Create features for TimestampedGeoJson
                features = []
                
                # Add point features
                for idx, row in user_data.iterrows():
                    try:
                        # Skip invalid coordinates
                        lat = float(row['latitude'])
                        lon = float(row['longitude'])
                        
                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                            continue
                        
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
                                'coordinates': [lon, lat]
                            },
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': 'red' if row.get('is_anomalous', False) else 'blue',
                                    'fillOpacity': 0.8,
                                    'stroke': 'true',
                                    'radius': 8 if row.get('is_anomalous', False) else 5
                                },
                                'popup': f"User: {user_id}<br>Time: {time_str}<br>Device: {row['device_id']}"
                            }
                        }
                        
                        features.append(point_feature)
                        
                    except Exception as e:
                        logger.warning(f"Error creating point feature: {e}")
                        continue
                
                # Add line features (trajectories)
                for i in range(len(features) - 1):
                    try:
                        point1 = features[i]['geometry']['coordinates']
                        point2 = features[i+1]['geometry']['coordinates']
                        time1 = features[i]['properties']['time']
                        
                        # Create line feature
                        line_feature = {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': [point1, point2]
                            },
                            'properties': {
                                'time': time1,
                                'style': {
                                    'color': '#3388ff',
                                    'weight': 3,
                                    'opacity': 0.8
                                }
                            }
                        }
                        
                        features.append(line_feature)
                        
                    except Exception as e:
                        logger.warning(f"Error creating line feature: {e}")
                        continue
                
                # Create TimestampedGeoJson for this user
                if features:
                    # Determine appropriate time settings
                    time_range = user_data['timestamp'].max() - user_data['timestamp'].min()
                    
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
                    
                    # Create a separate feature group for the time animation
                    time_fg = folium.FeatureGroup(name=f"Time Animation: {user_id}")
                    
                    # Add TimestampedGeoJson to the time feature group
                    TimestampedGeoJson(
                        {
                            'type': 'FeatureCollection',
                            'features': features
                        },
                        period=period,
                        duration=duration,
                        auto_play=False,
                        loop=False,
                        max_speed=10,
                        date_options='YYYY-MM-DD HH:mm:ss',
                        time_slider_drag_update=True
                    ).add_to(time_fg)
                    
                    # Add the time feature group to the map
                    time_fg.add_to(m)
            
        except Exception as e:
            logger.error(f"Error creating time-based visualization: {e}")
            logger.error(traceback.format_exc())
        
        # Add all feature groups to the map
        all_markers_fg.add_to(m)
        heatmap_fg.add_to(m)
        anomaly_fg.add_to(m)
        
        for user_fg in user_fgs.values():
            user_fg.add_to(m)
        
        # Add Search plugin using GeoJSON data
        try:
            search = Search(
                layer=all_markers_fg,
                geom_type="Point",
                placeholder="Search for users...",
                collapsed=False,
                search_label="user_id",  # Search by user_id property
                search_zoom=15,
                position='topleft'
            )
            
            m.add_child(search)
            logger.info("Added search plugin")
        except Exception as e:
            logger.error(f"Error adding search plugin: {e}")
            logger.error(traceback.format_exc())
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                   bottom: 50px; right: 50px; 
                   width: 180px; height: 160px; 
                   border: 2px solid grey; z-index: 9999; 
                   font-size: 14px;
                   background-color: white; 
                   padding: 10px;
                   border-radius: 5px;">
            <p style="margin-top: 0;"><b>Login Legend</b></p>
            <div>
                <i class="fa fa-user fa-1x" style="color:blue"></i>
                <span style="padding-left:5px;">Standard Login</span>
            </div>
            <div style="margin-top: 5px;">
                <i class="fa fa-exclamation-circle fa-1x" style="color:red"></i>
                <span style="padding-left:5px;">Anomalous Login</span>
            </div>
            <div style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
                <div style="height: 10px; width: 25px; background: linear-gradient(to right, blue, lime, yellow, orange, red); display: inline-block;"></div>
                <span style="vertical-align: middle; margin-left: 5px;">Anomaly Score</span>
            </div>
            <div style="margin-top: 5px;">
                <div style="width: 25px; height: 3px; background-color: #3388ff; display: inline-block;"></div>
                <span style="vertical-align: middle; margin-left: 5px;">User Path</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Helper info about search
        search_info_html = """
        <div style="position: fixed; 
                  top: 60px; left: 60px; 
                  width: 240px; 
                  border: 1px solid #ccc; 
                  z-index: 9999; 
                  font-size: 12px;
                  background-color: white; 
                  padding: 8px;
                  border-radius: 4px;">
            <p style="margin: 0 0 5px 0;"><b>Search Tips:</b></p>
            <ul style="margin: 0; padding-left: 20px;">
                <li>Type a user ID in the search box</li>
                <li>Select "Time Animation: [user]" to see movement</li>
                <li>Use the time slider controls to track movement</li>
            </ul>
        </div>
        """
        m.get_root().html.add_child(folium.Element(search_info_html))
        
        # Save the map
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
                alt_path = os.path.join(os.path.expanduser("~"), "anomaly_map_search.html")
                m.save(alt_path)
                logger.info(f"Map saved to alternate location: {alt_path}")
                return m
            except Exception as e2:
                logger.error(f"Failed to save to alternate location: {e2}")
                return m  # Return unsaved map as last resort
                
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    n = 100
    
    # Create sample data
    data = pd.DataFrame({
        'user_id': np.repeat(['user1', 'user2', 'user3', 'user4'], n//4),
        'timestamp': pd.date_range(start='2023-01-01', periods=n, freq='6H'),
        'latitude': np.random.uniform(40.7, 40.8, n),  # NYC area
        'longitude': np.random.uniform(-74.05, -73.95, n),
        'device_id': np.random.choice(['desktop', 'mobile', 'tablet'], n),
        'is_anomalous': np.random.choice([True, False], n, p=[0.1, 0.9]),
    })
    
    # Add anomaly scores
    data['anomaly_score'] = np.where(data['is_anomalous'], 
                                   np.random.uniform(0.7, 1.0, n),
                                   np.random.uniform(0.0, 0.3, n))
    
    # Create visualization
    try:
        result = visualize_with_folium_search(data)
        
        if result:
            print("Test completed successfully")
        else:
            print("Test failed")
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()