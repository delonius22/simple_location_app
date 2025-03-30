import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from datetime import datetime
import numpy as np
import json
from folium.plugins import TimestampedGeoJson

def visualize_user_locations(data_df, save_path='user_locations.html'):
    """
    Visualize user location data with anomaly highlighting using Folium.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame with columns: user_id, lat, long, device_id, timestamp, is_anomaly, anomaly_score
    save_path : str
        Path to save the HTML output
    
    Returns:
    --------
    folium.Map : The created map object
    """
    # Convert to DataFrame if it's not already
    if not isinstance(data_df, pd.DataFrame):
        data_df = pd.DataFrame(data_df)
    
    # Ensure timestamp is in datetime format
    if data_df['timestamp'].dtype != 'datetime64[ns]':
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    
    # Calculate map center based on mean coordinates
    center_lat = data_df['lat'].mean()
    center_lng = data_df['long'].mean()
    
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
            <b>Coordinates:</b> [{row['lat']:.5f}, {row['long']:.5f}]<br>
            <b>Anomaly:</b> {'Yes' if row['is_anomaly'] else 'No'}<br>
            <b>Anomaly Score:</b> {row['anomaly_score']:.4f}
        </div>
        """
        
        # Create marker with popup
        if row['is_anomaly']:
            # Red marker for anomalies
            marker = folium.Marker(
                location=[row['lat'], row['long']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='exclamation-circle', prefix='fa'),
                tooltip=f"Anomaly: User {row['user_id']}"
            )
            # Add to anomaly cluster
            marker.add_to(anomaly_cluster)
        else:
            # Blue marker for standard logins
            marker = folium.Marker(
                location=[row['lat'], row['long']],
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
        heatmap_data.append([row['lat'], row['long'], weight])
    
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
        color = 'red' if row['is_anomaly'] else 'blue'
        radius = 8 if row['is_anomaly'] else 5
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['long'], row['lat']]
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
        time_data,
        period='PT1H',  # Update period (1 hour)
        duration='PT10M',  # Duration to show points (10 minutes)
        transition_time=200,
        auto_play=False,
        loop=False,
        name='Login Timeline'
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

def create_sample_data(n_users=5, n_points=100, anomaly_ratio=0.15):
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    data = []
    for user_id in range(1, n_users+1):
        # Create center point for this user
        center_lat = np.random.uniform(35, 45)
        center_lng = np.random.uniform(-120, -70)
        
        # Generate standard locations around the center
        n_standard = int(n_points * (1 - anomaly_ratio))
        for i in range(n_standard):
            # Standard points close to center
            lat = center_lat + np.random.normal(0, 0.05)
            lng = center_lng + np.random.normal(0, 0.05)
            
            # Random timestamp within the last month
            days_ago = np.random.randint(0, 30)
            hours_ago = np.random.randint(0, 24)
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=days_ago, hours=hours_ago)
            
            device_id = f"device_{user_id}_{np.random.randint(1, 4)}"  # 1-3 devices per user
            
            # Low anomaly score for standard points
            anomaly_score = np.random.uniform(0, 0.3)
            
            data.append({
                'user_id': f"user_{user_id}",
                'lat': lat,
                'long': lng,
                'device_id': device_id,
                'timestamp': timestamp,
                'is_anomaly': False,
                'anomaly_score': anomaly_score
            })
        
        # Generate anomalous locations far from center
        n_anomalies = int(n_points * anomaly_ratio)
        for i in range(n_anomalies):
            # Anomalous points far from center
            lat = center_lat + np.random.normal(0, 0.5)  # More spread
            lng = center_lng + np.random.normal(0, 0.5)
            
            # Random timestamp
            days_ago = np.random.randint(0, 30)
            hours_ago = np.random.randint(0, 24)
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=days_ago, hours=hours_ago)
            
            # For anomalies, sometimes use an unusual device
            if np.random.random() < 0.7:
                device_id = f"unusual_device_{np.random.randint(100, 999)}"
            else:
                device_id = f"device_{user_id}_{np.random.randint(1, 4)}"
            
            # High anomaly score for anomalous points
            anomaly_score = np.random.uniform(0.7, 1.0)
            
            data.append({
                'user_id': f"user_{user_id}",
                'lat': lat,
                'long': lng,
                'device_id': device_id,
                'timestamp': timestamp,
                'is_anomaly': True,
                'anomaly_score': anomaly_score
            })
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = create_sample_data()
    
    # Visualize the data
    visualize_user_locations(sample_data, "user_location_visualization.html")
    