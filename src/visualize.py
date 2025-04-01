import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson,MarkerCluster, HeatMap
import os
import logging
import traceback
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_user_locations(data_df, save_path='user_locations.html'):
    """Create a robust visualization of login data with error handling."""
    logger.info("Starting visualization process")
    
    try:
        # Convert to DataFrame if not already
        if not isinstance(data_df, pd.DataFrame):
            data_df = pd.DataFrame(data_df)
            logger.info(f"Converted input to DataFrame with {len(data_df)} rows")
        
        # Check if DataFrame is empty
        if len(data_df) == 0:
            logger.error("Input DataFrame is empty")
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
            # Create a simple timestamp column for visualization
            data_df['timestamp'] = pd.Timestamp.now()
        
        # Add missing columns with default values if needed
        if 'is_anomalous' not in data_df.columns:
            data_df['is_anomalous'] = False
            logger.info("Added default 'is_anomalous' column (all False)")
        
        if 'anomaly_score' not in data_df.columns:
            data_df['anomaly_score'] = 0.0
            logger.info("Added default 'anomaly_score' column (all 0.0)")
        
        # Make sure latitude and longitude are numeric and valid
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
            logger.info(f"Map center calculated: [{center_lat}, {center_lon}]")
        except Exception as e:
            logger.error(f"Error calculating map center: {e}")
            center_lat, center_lon = 0, 0  # Default to null island
        
        # Create map with error handling
        try:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10, 
                           tiles='CartoDB positron')
            logger.info("Base map created successfully")
        except Exception as e:
            logger.error(f"Error creating map: {e}")
            return None
        
        # Add marker clusters - separate try/except for each component
        try:
            standard_cluster = MarkerCluster(name="Standard Logins").add_to(m)
            anomaly_cluster = MarkerCluster(name="Anomalous Logins").add_to(m)
            logger.info("Added marker clusters to map")
        except Exception as e:
            logger.error(f"Error adding marker clusters: {e}")
            # Continue without clusters
            standard_cluster = m
            anomaly_cluster = m
        
        # Process each data point for markers
        markers_added = 0
        try:
            for idx, row in data_df.iterrows():
                try:
                    # Skip rows with NaN coordinates
                    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                        continue
                    
                    # Ensure numeric lat/lon values
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    
                    # Check for valid coordinate range
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        logger.warning(f"Skipping row {idx} with invalid coordinates: [{lat}, {lon}]")
                        continue
                    
                    # Convert timestamp to string safely
                    if isinstance(row['timestamp'], pd.Timestamp):
                        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        time_str = str(row['timestamp'])
                    
                    # Format user_id and device_id as strings
                    user_id = str(row['user_id'])
                    device_id = str(row['device_id'])
                    
                    # Create popup content
                    popup_content = f"""
                    <div style="font-family: sans-serif; width: 200px;">
                        <h4 style="margin-bottom: 10px;">User Login Details</h4>
                        <b>User ID:</b> {user_id}<br>
                        <b>Device ID:</b> {device_id}<br>
                        <b>Timestamp:</b> {time_str}<br>
                        <b>Coordinates:</b> [{lat:.5f}, {lon:.5f}]<br>
                        <b>Anomaly:</b> {'Yes' if row.get('is_anomalous', False) else 'No'}<br>
                        <b>Anomaly Score:</b> {float(row.get('anomaly_score', 0.0)):.4f}
                    </div>
                    """
                    
                    # Create marker
                    if row.get('is_anomalous', False):
                        icon = folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
                    else:
                        icon = folium.Icon(color='blue', icon='user', prefix='fa')
                    
                    marker = folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=icon,
                        tooltip=f"{'Anomaly' if row.get('is_anomalous', False) else 'Standard'}: User {user_id}"
                    )
                    
                    # Add to appropriate cluster
                    if row.get('is_anomalous', False):
                        marker.add_to(anomaly_cluster)
                    else:
                        marker.add_to(standard_cluster)
                    
                    markers_added += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing marker for row {idx}: {e}")
                    # Skip this row and continue
                    continue
            
            logger.info(f"Added {markers_added} markers to map")
        except Exception as e:
            logger.error(f"Error processing markers: {e}")
            # Continue with whatever markers were added
        
        # Create a separate feature group for the heatmap (IMPORTANT: no popups here)
        try:
            # Create a feature group for the heatmap
            heatmap_group = folium.FeatureGroup(name="Anomaly Heatmap")
            
            # Prepare heatmap data (completely separate from markers)
            heatmap_data = []
            for _, row in data_df.iterrows():
                try:
                    # Only use rows with valid coordinates
                    if (pd.isna(row['latitude']) or pd.isna(row['longitude']) or 
                        not (-90 <= float(row['latitude']) <= 90) or 
                        not (-180 <= float(row['longitude']) <= 180)):
                        continue
                    
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    score = float(row.get('anomaly_score', 0.0))
                    
                    # Add to heatmap data (just coordinates and weight, no popups)
                    heatmap_data.append([lat, lon, score * 10])  # Scale score for visibility
                    
                except Exception as e:
                    continue  # Skip problem rows
            
            # Only add heatmap if we have data
            if heatmap_data:
                # Create the heatmap
                HeatMap(
                    data=heatmap_data,  # Just coordinates and weights
                    radius=15,
                    gradient={0.1: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1: 'red'},
                    min_opacity=0.5,
                    max_zoom=1
                ).add_to(heatmap_group)
                
                # Add the heatmap group to the map
                heatmap_group.add_to(m)
                
                logger.info(f"Added heatmap with {len(heatmap_data)} points")
            else:
                logger.warning("No valid data for heatmap")
                
        except Exception as e:
            logger.error(f"Error adding heatmap: {e}")
            logger.error(traceback.format_exc())
            # Continue without heatmap
            
        # Add TimestampedGeoJson visualization
        try:
            # Prepare features for TimestampedGeoJson
            features = []
            
            # Sort data by timestamp
            sorted_df = data_df.sort_values('timestamp')
            
            # Process each data point
            for idx, row in sorted_df.iterrows():
                try:
                    # Skip invalid coordinates
                    if (pd.isna(row['latitude']) or pd.isna(row['longitude']) or 
                        not (-90 <= float(row['latitude']) <= 90) or 
                        not (-180 <= float(row['longitude']) <= 180)):
                        continue
                    
                    # Extract properties
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    user_id = str(row['user_id'])
                    device_id = str(row['device_id'])
                    is_anomaly = bool(row.get('is_anomalous', False))
                    anomaly_score = float(row.get('anomaly_score', 0.0))
                    
                    # Format timestamp
                    if isinstance(row['timestamp'], pd.Timestamp):
                        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        time_str = str(row['timestamp'])
                    
                    # Create feature
                    feature = {
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
                    
                    features.append(feature)
                    
                except Exception as e:
                    logger.warning(f"Error processing time feature for row {idx}: {e}")
                    continue
            
            # Create TimestampedGeoJson if we have features
            if features:
                # Determine time range
                start_time = sorted_df['timestamp'].min()
                end_time = sorted_df['timestamp'].max()
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
                
                # Create TimestampedGeoJson
                time_slider = TimestampedGeoJson(
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
                
                # Add to map
                time_slider.add_to(m)
                logger.info(f"Added TimestampedGeoJson with {len(features)} features")
            else:
                logger.warning("No valid features for TimestampedGeoJson")
                
        except Exception as e:
            logger.error(f"Error adding TimestampedGeoJson: {e}")
            logger.error(traceback.format_exc())
            # Continue without time slider
        
        # Add layer control
        try:
            folium.LayerControl().add_to(m)
            logger.info("Added layer control")
        except Exception as e:
            logger.error(f"Error adding layer control: {e}")
        
        # Add legend
        try:
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 180px; height: 170px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color: white; padding: 10px;
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
                <div style="margin-top: 10px;">
                    <i class="fa fa-clock-o fa-1x"></i>
                    <span style="padding-left:5px;">Time Animation</span>
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            logger.info("Added legend")
        except Exception as e:
            logger.error(f"Error adding legend: {e}")
        
        # Save map - critical step
        try:
            # Get directory and ensure it exists
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created directory: {save_dir}")
                
            m.save(save_path)
            logger.info(f"Map saved successfully to {save_path}")
            
            # Verify file exists
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                logger.info(f"Verified file creation: {save_path} ({file_size} bytes)")
            else:
                logger.warning(f"File creation verification failed: {save_path} does not exist")
        except Exception as e:
            logger.error(f"Error saving map: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try a directory that's guaranteed to be writable
            try:
                # Try a different directory (home directory or current directory)
                alt_path = os.path.join(os.path.expanduser("~"), "anomaly_map_fallback.html")
                m.save(alt_path)
                logger.info(f"Map saved to alternate location: {alt_path}")
                return m, alt_path  # Return both the map and the alternate path
            except Exception as e2:
                logger.error(f"Failed to save to alternate location: {e2}")
                # As a last resort, return the map object without saving
                return m
        
        return m
        
    except Exception as e:
        logger.error(f"Unexpected error in visualization process: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        'user_id': np.repeat(['user1', 'user2'], n//2),
        'timestamp': pd.date_range(start='2023-01-01', periods=n, freq='6H'),
        'latitude': np.random.uniform(40.7, 40.8, n),  # NYC area
        'longitude': np.random.uniform(-74.05, -73.95, n),
        'device_id': np.random.choice(['device1', 'device2', 'device3'], n),
        'is_anomalous': np.random.choice([True, False], n, p=[0.1, 0.9]),
    })
    
    # Add anomaly scores
    data['anomaly_score'] = np.where(data['is_anomalous'], 
                                    np.random.uniform(0.7, 1.0, n),
                                    np.random.uniform(0.0, 0.3, n))
    
    # Create visualization
    try:
        result = visualize_user_locations(data)
        if result:
            print("Test completed! Check the generated HTML file.")
        else:
            print("Test failed! No visualization was created.")
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()
    # visualize_user_locations(sample_data, "user_location_visualization.html")
    