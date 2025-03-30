"""Example usage of Banking Anomaly Detection Lite."""

import json
import pandas as pd
import logging
from datetime import datetime
from sqlalchemy.orm import Session
import folium
from folium.plugins import MarkerCluster, HeatMap
from folium.plugins import TimestampedGeoJson
from src.db.base import SessionLocal, init_db
from src.services.processor import UserLoginProcessor
from IPython.display import display, HTML

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_example():
    """Run example of anomaly detection with sample data."""
    logger.info("Initializing database...")
    init_db()  # Create tables
    
    # Create database session
    db_session = SessionLocal()
    
    try:
        # Load sample data
        logger.info("Loading sample data...")
        with open('tests/sample_data.json', 'r') as f:
            data = json.load(f)
        
        login_df = pd.DataFrame(data['login_events'])
        
        # Create processor instance
        processor = UserLoginProcessor(db=db_session)
        
        # Process login data to build profiles
        logger.info("Processing login data...")
        result = processor.process_login_data(login_df)
        logger.info(f"Processed {result['users_processed']} users")
        
        logger.info("Analyzing test login...")
        analysis = processor.analyze_login(login_df)
        logger.info(f"User ID: {login_df['user_id']}")
        logger.info(f"Timestamp: {login_df['timestamp']}")
        logger.info(f"Latitude: {login_df['latitude']}")
        logger.info(f"Longitude: {login_df['longitude']}")
        logger.info(f"Device ID: {login_df['device_id']}")
        logger.info(f"IP Address: {login_df['ip_address']}")
        logger.info(f"Anomalous: {analysis['is_anomalous']}")
        logger.info(f"Score: {analysis['combined_score']}")
        logger.info(f"Anomaly scores: {analysis['anomaly_scores']}")
        
    finally:
        db_session.close()
    
def visualize(data_df:pd.DataFrame,save_path="user_location_visualization.html"):
    """Visualize the analysis results."""
    # Convert to DataFrame if it's not already
    if not isinstance(data_df, pd.DataFrame):
        data_df = pd.DataFrame(data_df)
    
    # Ensure timestamp is in datetime format
    if data_df['timestamp'].dtype != 'datetime64[ns]':
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    
    # Calculate map center based on mean coordinates
    center_lat = data_df['latitude'].mean()
    center_lng = data_df['longitude'].mean()

    # Create base map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=10, 
                   tiles='CartoDB positron')
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Create marker clusters for standard and anomalous logins
    standard_cluster = MarkerCluster(name="Standard Logins", overlay=True).add_to(m)
    anomaly_cluster = MarkerCluster(name="Anomalous Logins", overlay=True).add_to(m)
    
    # Create user-specific layers
    user_layers = {}
    for user_id in data_df['user_id'].unique():
        user_layer = folium.FeatureGroup(name=f"User {user_id}", show=False)
        user_layers[user_id] = user_layer
        m.add_child(user_layer)
    
    # Add markers for each login
    for idx, row in data_df.iterrows():
        # Format popup content
        popup_content = f"""
        <div style="font-family: sans-serif; width: 200px;">
            <h4 style="margin-bottom: 10px;">User Login Details</h4>
            <b>User ID:</b> {row['user_id']}<br>
            <b>Device ID:</b> {row['device_id']}<br>
            <b>Timestamp:</b> {row['timestamp']}<br>
            <b>Coordinates:</b> [{row['latitude']:.5f}, {row['longitude']:.5f}]<br>
            <b>Anomaly:</b> {'Yes' if row['is_anomalous'] else 'No'}<br>
            <b>Anomaly Score:</b> {row['anomaly_score']:.4f}
        </div>
        """
        
        # Create marker with popup
        if row['is_anomalous']:
            # Red marker for anomalies
            marker = folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='exclamation-circle', prefix='fa'),
                tooltip=f"Anomaly: User {row['user_id']}"
            )
            # Add to anomaly cluster
            marker.add_to(anomaly_cluster)
        else:
            # Blue marker for standard logins
            marker = folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='blue', icon='user', prefix='fa'),
                tooltip=f"Standard: User {row['user_id']}"
            )
            # Add to standard cluster
            marker.add_to(standard_cluster)
        
        # Also add to user-specific layer
        marker.add_to(user_layers[row['user_id']])
    
    # Add heatmap colored by anomaly score
    heatmap_data = []
    for idx, row in data_df.iterrows():
        # For heatmap, weight by anomaly score (higher score = more intense)
        weight = float(row['anomaly_score']) * 10  # Adjust multiplier as needed
        heatmap_data.append([row['latitude'], row['longitude'], weight])
    
    # Create and add heatmap layer
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
    
    # Add time-based visualization
    # Prepare GeoJSON for timestamped layer
    features = []
    
    for idx, row in data_df.iterrows():
        # Convert timestamp to Unix time (milliseconds)
        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Choose color based on anomaly status
        color = 'red' if row['is_anomalous'] else 'blue'
        radius = 8 if row['is_anomalous'] else 5
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['longitude'], row['latitude']]
            },
            'properties': {
                'time': time_str,
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': 0.8,
                    'stroke': True,
                    'radius': radius
                },
                'style': {'weight': 0},
                'popup': f"User: {row['user_id']}, Time: {time_str}"
            }
        }
        features.append(feature)
    
    # Create TimestampedGeoJson layer
    time_data = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    time_layer = TimestampedGeoJson(
        time_data,  # Pass the GeoJSON object directly
        period='PT1H',  # Update period (1 hour)
        duration='PT10M',  # Duration to show points (10 minutes)
        transition_time=200,
        auto_play=False,
        loop=False
    )
    m.add_child(time_layer)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 160px; 
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
        <div style="font-size: 12px; color: #666; margin-top: 10px;">Use timeline to see login patterns over time</div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save(save_path)
    print(f"Map saved to {save_path}")
    
    return m




if __name__ == "__main__":
    run_example()
    from sqlalchemy import create_engine
    
    # Create engine first to properly handle datetime objects
    engine = create_engine('sqlite:///login_anomaly_detection.db')
    data_df = pd.read_sql_table('login_events', engine, parse_dates=['timestamp'])
    display(data_df)
    # map = visualize(data_df)
    # print(map)