from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row, gridplot
from bokeh.models import (ColumnDataSource, HoverTool, ColorBar, LinearColorMapper, 
                         DatetimeTickFormatter, Panel, Tabs, Legend, LegendItem, 
                         Range1d, MultiChoice, CustomJS, Div, Button, Select)
from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import Viridis256, Category10
from bokeh.transform import linear_cmap
import pandas as pd
import numpy as np
import os
import logging
import traceback
from datetime import datetime, timedelta
import pyproj
from pyproj import Transformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def lonlat_to_web_mercator(lon, lat):
    """Convert longitude/latitude to Web Mercator coordinates"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def visualize_with_bokeh(data_df, save_path="bokeh_user_location_analysis.html"):
    """
    Create an interactive visualization of user login locations and anomalies using Bokeh.
    
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
    bool
        Success status
    """
    logger.info("Starting Bokeh visualization process")
    
    try:
        # Convert to DataFrame if not already
        if not isinstance(data_df, pd.DataFrame):
            data_df = pd.DataFrame(data_df)
            logger.info(f"Converted input to DataFrame with {len(data_df)} rows")
        
        # Check if DataFrame is empty
        if len(data_df) == 0:
            logger.error("Input DataFrame is empty")
            return False
        
        # Ensure required columns exist
        required_columns = ['user_id', 'latitude', 'longitude', 'timestamp']
        for col in required_columns:
            if col not in data_df.columns:
                logger.error(f"Required column '{col}' missing from input data")
                return False
        
        # Add missing columns with default values if needed
        if 'device_id' not in data_df.columns:
            data_df['device_id'] = 'unknown'
        
        if 'is_anomalous' not in data_df.columns:
            data_df['is_anomalous'] = False
        
        if 'anomaly_score' not in data_df.columns:
            data_df['anomaly_score'] = 0.0
        
        # Ensure timestamp is datetime
        try:
            if pd.api.types.is_datetime64_any_dtype(data_df['timestamp']) == False:
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                logger.info("Converted timestamp column to datetime")
        except Exception as e:
            logger.error(f"Error converting timestamps: {e}")
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
                return False
        except Exception as e:
            logger.error(f"Error processing coordinates: {e}")
            return False
        
        # Sort by timestamp for time-based analysis
        data_df = data_df.sort_values('timestamp')
        
        # Calculate Web Mercator coordinates for the map
        data_df['x'], data_df['y'] = zip(*data_df.apply(
            lambda row: lonlat_to_web_mercator(row['longitude'], row['latitude']), 
            axis=1
        ))
        
        # Format the timestamp for display
        data_df['timestamp_str'] = data_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a label for anomaly status
        data_df['anomaly_status'] = data_df['is_anomalous'].apply(
            lambda x: 'Anomalous' if x else 'Normal'
        )
        
        # Create data source for normal and anomalous logins
        normal_df = data_df[~data_df['is_anomalous']]
        anomaly_df = data_df[data_df['is_anomalous']]
        
        normal_source = ColumnDataSource(normal_df)
        anomaly_source = ColumnDataSource(anomaly_df)
        
        # Set output file
        output_file(save_path, title="User Location and Anomaly Analysis")
        
        # Create map plot
        map_options = {
            'x_axis_type': 'mercator',
            'y_axis_type': 'mercator',
            'width': 800,
            'height': 500,
            'title': 'User Login Locations',
            'tools': 'pan,wheel_zoom,box_zoom,reset,save,hover',
            'active_scroll': 'wheel_zoom'
        }
        map_plot = figure(**map_options)
        
        # Add tile provider (map background)
        tile_provider = get_provider(Vendors.CARTODBPOSITRON)
        map_plot.add_tile(tile_provider)
        
        # Add hover tool for map
        map_hover = HoverTool(
            tooltips=[
                ("User", "@user_id"),
                ("Device", "@device_id"),
                ("Time", "@timestamp_str"),
                ("Status", "@anomaly_status"),
                ("Anomaly Score", "@anomaly_score{0.0000}")
            ]
        )
        map_plot.add_tools(map_hover)
        
        # Plot normal logins
        normal_circles = map_plot.circle(
            x='x', y='y', 
            source=normal_source,
            size=8, 
            color='blue', 
            alpha=0.6,
            legend_label="Normal Logins"
        )
        
        # Plot anomalous logins
        anomaly_circles = map_plot.circle(
            x='x', y='y', 
            source=anomaly_source,
            size=12, 
            color='red', 
            alpha=0.7,
            legend_label="Anomalous Logins"
        )
        
        # Position the legend
        map_plot.legend.location = "top_left"
        map_plot.legend.click_policy = "hide"
        
        # Create time series plot for anomaly scores
        time_plot = figure(
            width=800, 
            height=300, 
            title="Anomaly Scores Over Time",
            x_axis_type='datetime',
            tools='pan,wheel_zoom,box_zoom,reset,save,hover'
        )
        
        # Format the time axis
        time_plot.xaxis.formatter = DatetimeTickFormatter(
            hours=["%H:%M"],
            days=["%m/%d"],
            months=["%m/%Y"],
            years=["%Y"]
        )
        
        # Add hover tool for time plot
        time_hover = HoverTool(
            tooltips=[
                ("User", "@user_id"),
                ("Time", "@timestamp_str"),
                ("Status", "@anomaly_status"),
                ("Score", "@anomaly_score{0.0000}")
            ]
        )
        time_plot.add_tools(time_hover)
        
        # Create a color mapper for anomaly scores
        mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
        
        # Plot each user with a different color
        users = data_df['user_id'].unique()
        colors = Category10[10][:len(users)]
        user_color_map = dict(zip(users, colors))
        
        # Add a separate line for each user
        for i, user in enumerate(users):
            user_data = data_df[data_df['user_id'] == user]
            user_source = ColumnDataSource(user_data)
            
            # Plot the user's timeline
            time_plot.circle(
                x='timestamp',
                y='anomaly_score',
                source=user_source,
                size=8,
                color=user_color_map[user],
                alpha=0.7,
                legend_label=f"User {user}"
            )
        
        # Add a threshold line if there are anomalies
        if len(anomaly_df) > 0:
            min_anomaly_score = anomaly_df['anomaly_score'].min()
            time_plot.line(
                x=[data_df['timestamp'].min(), data_df['timestamp'].max()],
                y=[min_anomaly_score, min_anomaly_score],
                line_color='red',
                line_dash='dashed',
                line_width=2,
                legend_label="Anomaly Threshold"
            )
        
        # Configure the time plot legend
        time_plot.legend.location = "top_left"
        time_plot.legend.click_policy = "hide"
        time_plot.xaxis.axis_label = "Time"
        time_plot.yaxis.axis_label = "Anomaly Score"
        
        # Create a color map plot for showing anomaly density over time
        heat_plot = figure(
            width=800,
            height=200,
            title="Anomaly Score Density Over Time",
            x_axis_type='datetime',
            y_range=(0, 1),
            tools='pan,wheel_zoom,box_zoom,reset,save'
        )
        
        # Format the time axis
        heat_plot.xaxis.formatter = DatetimeTickFormatter(
            hours=["%H:%M"],
            days=["%m/%d"],
            months=["%m/%Y"],
            years=["%Y"]
        )
        
        # Create a color map for the anomaly scores
        color_mapper = linear_cmap(
            field_name='anomaly_score', 
            palette=Viridis256, 
            low=data_df['anomaly_score'].min(),
            high=data_df['anomaly_score'].max()
        )
        
        # Plot all points with color based on anomaly score
        heat_points = heat_plot.circle(
            x='timestamp',
            y='anomaly_score',
            source=ColumnDataSource(data_df),
            size=10,
            color=color_mapper,
            alpha=0.7
        )
        
        # Add color bar
        color_bar = ColorBar(
            color_mapper=mapper,
            location=(0, 0),
            title="Anomaly Score",
            orientation='horizontal',
            width=8,
            height=200
        )
        heat_plot.add_layout(color_bar, 'right')
        
        # Add a div for statistics
        stats_html = f"""
        <div style="padding: 12px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9;">
            <h3 style="margin-top: 0; color: #2b5d8c;">Login Statistics Summary</h3>
            <ul>
                <li><b>Total Logins:</b> {len(data_df)}</li>
                <li><b>Normal Logins:</b> {len(normal_df)}</li>
                <li><b>Anomalous Logins:</b> {len(anomaly_df)}</li>
                <li><b>Date Range:</b> {data_df['timestamp'].min().strftime('%Y-%m-%d')} to {data_df['timestamp'].max().strftime('%Y-%m-%d')}</li>
                <li><b>Unique Users:</b> {len(users)}</li>
                <li><b>Unique Devices:</b> {data_df['device_id'].nunique()}</li>
            </ul>
            <p style="margin-bottom: 0; font-style: italic;">
                Anomaly Threshold: {anomaly_df['anomaly_score'].min() if len(anomaly_df) > 0 else 'N/A'}
            </p>
        </div>
        """
        stats_div = Div(text=stats_html, width=800)
        
        # Create user analysis tabs
        user_tabs = []
        
        for user in users:
            user_data = data_df[data_df['user_id'] == user]
            user_normal = user_data[~user_data['is_anomalous']]
            user_anomalies = user_data[user_data['is_anomalous']]
            
            # Create user map
            user_map = figure(
                x_axis_type='mercator',
                y_axis_type='mercator',
                width=400,
                height=400,
                title=f"User {user} Login Locations",
                tools='pan,wheel_zoom,box_zoom,reset,save,hover'
            )
            
            # Add tile provider
            user_map.add_tile(tile_provider)
            
            # Add normal and anomalous points
            if len(user_normal) > 0:
                user_map.circle(
                    x='x', y='y',
                    source=ColumnDataSource(user_normal),
                    size=8,
                    color='blue',
                    alpha=0.6,
                    legend_label="Normal"
                )
                
            if len(user_anomalies) > 0:
                user_map.circle(
                    x='x', y='y',
                    source=ColumnDataSource(user_anomalies),
                    size=12,
                    color='red',
                    alpha=0.7,
                    legend_label="Anomalous"
                )
            
            # Add legend
            user_map.legend.location = "top_left"
            user_map.legend.click_policy = "hide"
            
            # Create user timeline
            user_time = figure(
                width=400,
                height=300,
                title=f"User {user} Anomaly Score Timeline",
                x_axis_type='datetime',
                tools='pan,wheel_zoom,box_zoom,reset,save,hover'
            )
            
            # Format the time axis
            user_time.xaxis.formatter = DatetimeTickFormatter(
                hours=["%H:%M"],
                days=["%m/%d"],
                months=["%m/%Y"],
                years=["%Y"]
            )
            
            # Add time series data
            user_time.line(
                x='timestamp',
                y='anomaly_score',
                source=ColumnDataSource(user_data),
                line_width=2,
                color=user_color_map[user]
            )
            
            user_time.circle(
                x='timestamp',
                y='anomaly_score',
                source=ColumnDataSource(user_data),
                size=8,
                color=user_color_map[user],
                alpha=0.7
            )
            
            # If there are anomalies, highlight them and add threshold
            if len(user_anomalies) > 0:
                user_time.circle(
                    x='timestamp',
                    y='anomaly_score',
                    source=ColumnDataSource(user_anomalies),
                    size=12,
                    color='red',
                    alpha=0.7
                )
                
                # Add threshold line
                min_anomaly_score = user_anomalies['anomaly_score'].min()
                user_time.line(
                    x=[user_data['timestamp'].min(), user_data['timestamp'].max()],
                    y=[min_anomaly_score, min_anomaly_score],
                    line_color='red',
                    line_dash='dashed',
                    line_width=2
                )
            
            # User statistics
            user_stats_html = f"""
            <div style="padding: 12px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9;">
                <h3 style="margin-top: 0; color: #2b5d8c;">User {user} Statistics</h3>
                <ul>
                    <li><b>Total Logins:</b> {len(user_data)}</li>
                    <li><b>Normal Logins:</b> {len(user_normal)}</li>
                    <li><b>Anomalous Logins:</b> {len(user_anomalies)}</li>
                    <li><b>Average Anomaly Score:</b> {user_data['anomaly_score'].mean():.4f}</li>
                    <li><b>First Login:</b> {user_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><b>Last Login:</b> {user_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    <li><b>Unique Devices:</b> {user_data['device_id'].nunique()}</li>
                </ul>
            </div>
            """
            user_stats_div = Div(text=user_stats_html, width=400)
            
            # Create user tab layout
            user_layout = column(
                row(user_map, column(user_stats_div, user_time))
            )
            
            # Add to tabs
            user_tabs.append(Panel(child=user_layout, title=f"User {user}"))
        
        # Create tabs layout
        tabs = Tabs(tabs=user_tabs)
        
        # Final layout
        layout = column(
            stats_div, 
            map_plot, 
            time_plot, 
            heat_plot,
            tabs
        )
        
        # Save the visualization
        try:
            # Get directory and ensure it exists
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created directory: {save_dir}")
            
            save(layout, save_path, title="User Location and Anomaly Analysis")
            
            logger.info(f"Visualization saved to {save_path}")
            
            # Verify file exists
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                logger.info(f"Verified file creation: {save_path} ({file_size} bytes)")
                return True
            else:
                logger.warning(f"File creation verification failed: {save_path} does not exist")
                return False
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try alternate location
            try:
                alt_path = os.path.join(os.path.expanduser("~"), "bokeh_visualization_fallback.html")
                save(layout, alt_path, title="User Location and Anomaly Analysis")
                logger.info(f"Visualization saved to alternate location: {alt_path}")
                return True
            except Exception as e2:
                logger.error(f"Failed to save to alternate location: {e2}")
                return False
                
    except Exception as e:
        logger.error(f"Unexpected error in visualization process: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Simple test with synthetic data
    print("Creating test visualization...")
    
    from sqlalchemy import create_engine
    
    # Create engine first to properly handle datetime objects
    engine = create_engine('sqlite:///login_anomaly_detection.db')
    data_df = pd.read_sql_table('login_events', engine, parse_dates=['timestamp'])
   
    
    # Visualize the data
    
    try:
        success = visualize_with_bokeh(data_df)
        if success:
            print("Visualization created successfully!")
        else:
            print("Failed to create visualization")
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()