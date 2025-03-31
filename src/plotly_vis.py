import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_locations_and_anomalies(data_df, save_path="user_location_analysis.html"):
    """
    Create an interactive visualization of user login locations and anomalies over time using Plotly.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing login data with columns:
        - user_id: User identifier
        - latitude: Latitude coordinate
        - longitude: Longitude coordinate
        - timestamp: Login timestamp
        - device_id: Device identifier
        - is_anomalous: Boolean flag for anomalies (optional)
        - anomaly_score: Numeric anomaly score (optional)
    
    save_path : str
        Path to save the HTML visualization
        
    Returns:
    --------
    tuple
        (fig, save_path) - The Plotly figure object and path where it was saved
    """
    logger.info("Starting Plotly visualization process")
    
    try:
        # Convert to DataFrame if not already
        if not isinstance(data_df, pd.DataFrame):
            data_df = pd.DataFrame(data_df)
            logger.info(f"Converted input to DataFrame with {len(data_df)} rows")
        
        # Check if DataFrame is empty
        if len(data_df) == 0:
            logger.error("Input DataFrame is empty")
            return None, None
        
        # Ensure required columns exist
        required_columns = ['user_id', 'latitude', 'longitude', 'timestamp']
        for col in required_columns:
            if col not in data_df.columns:
                logger.error(f"Required column '{col}' missing from input data")
                return None, None
        
        # Add optional columns with default values if needed
        if 'device_id' not in data_df.columns:
            data_df['device_id'] = 'unknown'
            logger.info("Added default 'device_id' column (all 'unknown')")
            
        if 'is_anomalous' not in data_df.columns:
            data_df['is_anomalous'] = False
            logger.info("Added default 'is_anomalous' column (all False)")
        
        if 'anomaly_score' not in data_df.columns:
            data_df['anomaly_score'] = 0.0
            logger.info("Added default 'anomaly_score' column (all 0.0)")
        
        # Ensure timestamp is datetime
        try:
            if data_df['timestamp'].dtype != 'datetime64[ns]':
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                logger.info("Converted timestamp column to datetime")
        except Exception as e:
            logger.error(f"Error converting timestamps: {e}")
            # Create a simple timestamp column for visualization
            data_df['timestamp'] = pd.Timestamp.now()
        
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
                return None, None
        except Exception as e:
            logger.error(f"Error processing coordinates: {e}")
            return None, None
        
        # Sort by timestamp for time-based analysis
        data_df = data_df.sort_values('timestamp')
        
        # Create a copy of the dataframe with user-friendly labels for display
        display_df = data_df.copy()
        display_df['anomaly_label'] = display_df['is_anomalous'].apply(lambda x: 'Anomaly' if x else 'Normal')
        display_df['hover_text'] = display_df.apply(
            lambda row: f"User: {row['user_id']}<br>" +
                       f"Device: {row['device_id']}<br>" +
                       f"Time: {row['timestamp']}<br>" +
                       f"Anomaly Score: {row['anomaly_score']:.4f}",
            axis=1
        )
        
        # Create a multi-panel visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("User Login Locations", "Anomaly Scores Over Time"),
            specs=[[{"type": "scattergeo"}], [{"type": "scatter"}]],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # 1. Add map with login locations
        # Separate normal and anomalous logins
        normal_logins = display_df[~display_df['is_anomalous']]
        anomalous_logins = display_df[display_df['is_anomalous']]
        
        # Add normal login points
        if len(normal_logins) > 0:
            fig.add_trace(
                go.Scattergeo(
                    lat=normal_logins['latitude'],
                    lon=normal_logins['longitude'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.7
                    ),
                    text=normal_logins['hover_text'],
                    hoverinfo='text',
                    name='Normal Logins'
                ),
                row=1, col=1
            )
        
        # Add anomalous login points
        if len(anomalous_logins) > 0:
            fig.add_trace(
                go.Scattergeo(
                    lat=anomalous_logins['latitude'],
                    lon=anomalous_logins['longitude'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        opacity=0.8
                    ),
                    text=anomalous_logins['hover_text'],
                    hoverinfo='text',
                    name='Anomalous Logins'
                ),
                row=1, col=1
            )
        
        # 2. Add timeline of anomaly scores
        fig.add_trace(
            go.Scatter(
                x=display_df['timestamp'],
                y=display_df['anomaly_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=display_df['anomaly_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Anomaly Score",
                        x=1.02,
                        y=0.3,
                        len=0.3
                    )
                ),
                text=display_df['hover_text'],
                hoverinfo='text',
                name='Anomaly Score'
            ),
            row=2, col=1
        )
        
        # Add threshold line if anomalies exist
        if display_df['is_anomalous'].any():
            # Calculate the minimum anomaly score for anomalous points
            anomaly_threshold = display_df[display_df['is_anomalous']]['anomaly_score'].min()
            fig.add_trace(
                go.Scatter(
                    x=[display_df['timestamp'].min(), display_df['timestamp'].max()],
                    y=[anomaly_threshold, anomaly_threshold],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    name='Anomaly Threshold'
                ),
                row=2, col=1
            )
        
        # Calculate map center
        center_lat = float(display_df['latitude'].mean())
        center_lon = float(display_df['longitude'].mean())
        
        # Update geo layout
        fig.update_geos(
            projection_type="natural earth",
            showcoastlines=True,
            coastlinecolor="RebeccaPurple",
            showland=True,
            landcolor="LightGreen",
            showocean=True,
            oceancolor="LightBlue",
            showlakes=True,
            lakecolor="Blue",
            showrivers=True,
            rivercolor="Blue",
            showcountries=True,
            countrycolor="Black",
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=5  # Adjust based on data spread
        )
        
        # Update overall layout
        fig.update_layout(
            title_text='User Login Locations and Anomaly Detection',
            height=900,
            width=1000,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Update axes for the time series plot
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Anomaly Score', row=2, col=1)
        
        # Save HTML file
        try:
            # Get directory and ensure it exists
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created directory: {save_dir}")
                
            fig.write_html(
                save_path,
                full_html=True,
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'responsive': True}
            )
            
            logger.info(f"Visualization saved to {save_path}")
            
            # Verify file exists
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                logger.info(f"Verified file creation: {save_path} ({file_size} bytes)")
            else:
                logger.warning(f"File creation verification failed: {save_path} does not exist")
                
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try alternate location
            try:
                alt_path = os.path.join(os.path.expanduser("~"), "anomaly_visualization_fallback.html")
                fig.write_html(alt_path)
                logger.info(f"Visualization saved to alternate location: {alt_path}")
                save_path = alt_path
            except Exception as e2:
                logger.error(f"Failed to save to alternate location: {e2}")
        
        return fig, save_path
        
    except Exception as e:
        logger.error(f"Unexpected error in visualization process: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def create_user_cluster_analysis(data_df, save_path="user_cluster_analysis.html"):
    """
    Create a multi-panel visualization showing user clusters and anomalies over time.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing login data with required columns
    
    save_path : str
        Path to save the HTML visualization
        
    Returns:
    --------
    tuple
        (fig, save_path) - The Plotly figure object and path where it was saved
    """
    logger.info("Starting user cluster analysis visualization")
    
    try:
        # Process data and add any missing columns
        data_df = data_df.copy()
        
        # Ensure timestamp is datetime
        if data_df['timestamp'].dtype != 'datetime64[ns]':
            data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        # Add anomaly columns if not present
        if 'is_anomalous' not in data_df.columns:
            data_df['is_anomalous'] = False
        
        if 'anomaly_score' not in data_df.columns:
            data_df['anomaly_score'] = 0.0
        
        # Get unique users
        users = data_df['user_id'].unique()
        n_users = len(users)
        
        if n_users == 0:
            logger.error("No users found in data")
            return None, None
        
        logger.info(f"Analyzing data for {n_users} users")
        
        # Create figure with subplots: map + timeline
        fig = make_subplots(
            rows=2, cols=1, 
            specs=[[{"type": "scattergeo"}], [{"type": "scatter"}]],
            vertical_spacing=0.1,
            subplot_titles=("User Login Locations by Cluster", "Login Activity Timeline"),
            row_heights=[0.7, 0.3]
        )
        
        # Assign colors to users
        colorscale = px.colors.qualitative.Plotly
        user_colors = {user: colorscale[i % len(colorscale)] for i, user in enumerate(users)}
        
        # Add data for each user
        for user in users:
            user_data = data_df[data_df['user_id'] == user]
            
            # Skip users with no valid data
            if len(user_data) == 0:
                continue
                
            # Create hover information
            hover_text = user_data.apply(
                lambda row: f"User: {row['user_id']}<br>" +
                           f"Device: {row['device_id']}<br>" +
                           f"Time: {row['timestamp']}<br>" +
                           f"Anomaly: {'Yes' if row['is_anomalous'] else 'No'}<br>" +
                           f"Score: {row['anomaly_score']:.4f}",
                axis=1
            )
            
            # Color points based on anomaly status for this user
            marker_colors = user_data['is_anomalous'].apply(
                lambda x: 'red' if x else user_colors[user]
            )
            
            # Add user locations to map
            fig.add_trace(
                go.Scattergeo(
                    lat=user_data['latitude'],
                    lon=user_data['longitude'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=marker_colors,
                        opacity=0.7
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=f"User {user}"
                ),
                row=1, col=1
            )
            
            # Add user timeline
            fig.add_trace(
                go.Scatter(
                    x=user_data['timestamp'],
                    y=user_data['anomaly_score'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=marker_colors,
                        opacity=0.7
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=f"User {user}"
                ),
                row=2, col=1
            )
        
        # Calculate map center
        center_lat = float(data_df['latitude'].mean())
        center_lon = float(data_df['longitude'].mean())
        
        # Update geo layout
        fig.update_geos(
            projection_type="natural earth",
            showcoastlines=True,
            coastlinecolor="RebeccaPurple",
            showland=True,
            landcolor="LightGreen",
            showocean=True,
            oceancolor="LightBlue",
            showcountries=True,
            countrycolor="Black",
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=5  # Adjust based on data spread
        )
        
        # Update overall layout
        fig.update_layout(
            title_text='User Cluster Analysis and Anomaly Detection',
            height=900,
            width=1000,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Update axes for the time series plot
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Anomaly Score', row=2, col=1)
        
        # Save HTML file
        try:
            # Get directory and ensure it exists
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            fig.write_html(
                save_path,
                full_html=True,
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'responsive': True}
            )
            
            logger.info(f"Cluster analysis saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving cluster analysis: {str(e)}")
            
            # Try alternate location
            try:
                alt_path = os.path.join(os.path.expanduser("~"), "cluster_analysis_fallback.html")
                fig.write_html(alt_path)
                logger.info(f"Analysis saved to alternate location: {alt_path}")
                save_path = alt_path
            except Exception as e2:
                logger.error(f"Failed to save to alternate location: {e2}")
        
        return fig, save_path
        
    except Exception as e:
        logger.error(f"Unexpected error in cluster analysis: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

if __name__ == "__main__":
    # Simple test with synthetic data
    print("Creating test visualization...")
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    
    # Create timestamps with a gap to simulate different login sessions
    timestamps = pd.date_range(start='2023-01-01', periods=n//2, freq='6H').tolist()
    timestamps += pd.date_range(start='2023-02-01', periods=n//2, freq='6H').tolist()
    
    # Create synthetic data with multiple users and some anomalies
    data = pd.DataFrame({
        'user_id': np.random.choice(['user1', 'user2', 'user3', 'user4'], n),
        'timestamp': timestamps,
        'device_id': np.random.choice(['desktop', 'mobile', 'tablet'], n),
    })
    
    # Assign different geographic areas to different users
    for user_id in data['user_id'].unique():
        mask = data['user_id'] == user_id
        n_user = mask.sum()
        
        if user_id == 'user1':  # San Francisco area
            base_lat, base_lon = 37.7749, -122.4194
        elif user_id == 'user2':  # New York area
            base_lat, base_lon = 40.7128, -74.0060
        elif user_id == 'user3':  # Los Angeles area
            base_lat, base_lon = 34.0522, -118.2437
        else:  # Chicago area
            base_lat, base_lon = 41.8781, -87.6298
        
        # Add some randomness to the locations
        data.loc[mask, 'latitude'] = base_lat + np.random.normal(0, 0.05, n_user)
        data.loc[mask, 'longitude'] = base_lon + np.random.normal(0, 0.05, n_user)
    
    # Add a few anomalies (significantly different locations)
    anomaly_indices = np.random.choice(range(n), size=int(n*0.1), replace=False)
    data.loc[anomaly_indices, 'is_anomalous'] = True
    
    # For anomalies, shift location significantly
    for idx in anomaly_indices:
        if data.loc[idx, 'user_id'] == 'user1':
            data.loc[idx, 'latitude'] += np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)
            data.loc[idx, 'longitude'] += np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)
        elif np.random.random() < 0.5:  # For some users, change to a completely different location
            data.loc[idx, 'latitude'] = np.random.uniform(25, 45)
            data.loc[idx, 'longitude'] = np.random.uniform(-125, -70)
    
    # Add anomaly scores
    data['anomaly_score'] = np.where(data.get('is_anomalous', False), 
                                    np.random.uniform(0.7, 1.0, n),
                                    np.random.uniform(0.0, 0.3, n))
    
    # Create both visualizations
    try:
        fig1, path1 = visualize_locations_and_anomalies(data)
        if fig1:
            print(f"Main visualization created at: {path1}")
        else:
            print("Failed to create main visualization")
            
        fig2, path2 = create_user_cluster_analysis(data)
        if fig2:
            print(f"Cluster analysis created at: {path2}")
        else:
            print("Failed to create cluster analysis")
            
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()