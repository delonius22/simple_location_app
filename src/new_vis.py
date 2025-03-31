# visualization_robust.py

import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_robust(data_df, save_path="user_location_visualization.html"):
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
        
        # Process each data point
        markers_added = 0
        try:
            for idx, row in data_df.iterrows():
                try:
                    # Ensure numeric lat/lon values
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    
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
                        <b>Anomaly:</b> {'Yes' if row['is_anomalous'] else 'No'}<br>
                        <b>Anomaly Score:</b> {float(row['anomaly_score']):.4f}
                    </div>
                    """
                    
                    # Create marker
                    if row['is_anomalous']:
                        icon = folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
                    else:
                        icon = folium.Icon(color='blue', icon='user', prefix='fa')
                    
                    marker = folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=icon,
                        tooltip=f"{'Anomaly' if row['is_anomalous'] else 'Standard'}: User {user_id}"
                    )
                    
                    # Add to appropriate cluster
                    if row['is_anomalous']:
                        marker.add_to(anomaly_cluster)
                    else:
                        marker.add_to(standard_cluster)
                    
                    markers_added += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    # Skip this row and continue
                    continue
            
            logger.info(f"Added {markers_added} markers to map")
        except Exception as e:
            logger.error(f"Error processing data points: {e}")
            # Continue with whatever markers were added
        
        # Add heatmap for anomaly scores
        try:
            # Create heatmap data safely
            heatmap_data = []
            for idx, row in data_df.iterrows():
                try:
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    score = float(row['anomaly_score']) * 10  # Scale for visibility
                    heatmap_data.append([lat, lon, score])
                except Exception as e:
                    logger.warning(f"Error processing heatmap point {idx}: {e}")
                    continue
            
            # Add heatmap to map
            if heatmap_data:
                heatmap = HeatMap(
                    heatmap_data,
                    name="Anomaly Heatmap",
                    show=False,
                    radius=15,
                    gradient={0.1: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1: 'red'},
                    min_opacity=0.5,
                    max_zoom=1
                )
                m.add_child(heatmap)
                logger.info(f"Added heatmap with {len(heatmap_data)} points")
        except Exception as e:
            logger.error(f"Error adding heatmap: {e}")
            # Continue without heatmap
        
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
                        bottom: 50px; right: 50px; width: 180px; height: 140px; 
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
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            logger.info("Added legend")
        except Exception as e:
            logger.error(f"Error adding legend: {e}")
        
        # Save map - critical step
        try:
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
            except Exception as e2:
                logger.error(f"Failed to save to alternate location: {e2}")
                # As a last resort, return the map object without saving
                return m
        
        return m
        
    except Exception as e:
        logger.error(f"Unexpected error in visualization process: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Simple test with synthetic data
    print("Creating test visualization...")
    
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        'user_id': np.repeat(['user1', 'user2'], n//2),
        'timestamp': pd.date_range(start='2023-01-01', periods=n, freq='H'),
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
    visualize_robust(data)
    
    print("Test completed! Check the generated HTML file.")