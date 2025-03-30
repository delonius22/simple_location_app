import numpy as np 
import pandas as pd
from typing import Dict


def handle_antimeridian(location_data: pd.DataFrame) -> pd.DataFrame:
    """Adjust longitude values to handle the antimeridian issue.
    
    Args:
        location_data: DataFrame with latitude and longitude columns
        
    Returns:
        DataFrame with adjusted longitude values
    """
    result = location_data.copy()

    # Check if points likely cross the antimeridian (international date line at 180°/-180°)
    # If longitude range > 180°, points are probably on opposite sides of the antimeridian
    if len(result) >1 and (max(result['longitude']) - min(result['longitude'])) > 180:
        # Convert longitude less than 0 to their equivalent 0-360 vals
        result.loc[result['longitude'] < 0, 'longitude'] += 360

    return result


def calculate_geographic_mean_center(location_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate the geographic mean center of a set of locations.
    
    Args:
        location_data: DataFrame with latitude and longitude columns
        
    Returns:
        Dictionary with latitude and longitude of the mean center
    """

    if len(location_data) == 0:
        return {'latitude': 0.0, 'longitude': 0.0}
    
    location_data = handle_antimeridian(location_data)

    # convert latitude and longitude to radians
    lat_rad = np.radians(location_data['latitude'])
    lon_rad = np.radians(location_data['longitude'])

    # convert to cartesian coordinates
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # calculate the mean x, y, z coordinates
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_z = np.mean(z)

    # convert mean x, y, z coordinates back to latitude and longitude
    mean_lon = np.arctan2(mean_y, mean_x)
    hyp = np.sqrt(mean_x**2 + mean_y**2)
    mean_lat = np.arctan2(mean_z, hyp)

    # convert to degrees
    central_lat = np.degrees(mean_lat)
    central_lon = np.degrees(mean_lon)

    #adjust longitude back to -180 to 180 range if necessary
    if central_lon > 180:
        central_lon -= 360

    return {'latitude': central_lat, 'longitude': central_lon}


def haversine_distance(lat1:float,lon1:float,lat2:float,lon2:float) -> float:
    """Calculate the Haversine distance between two points.
    
    Args:
        lat1: Latitude of the first point
        lon1: Longitude of the first point
        lat2: Latitude of the second point
        lon2: Longitude of the second point
        
    Returns:
        Haversine distance between the two points
    """
    R = 6371  # radius of the Earth in km
    phi1 = np.radians(lat1) # convert to radians
    phi2 = np.radians(lat2) # convert to radians
    delta_phi = np.radians(lat2 - lat1) # convert to radians for calculating distance of latitude
    delta_lambda = np.radians(lon2 - lon1) # convert to radians for calculating distance of longitude

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2 # calculate the square of half the chord length
    res = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))) # calculate the distance in km
    return np.round(res, 2) # return the distance rounded to 2 decimal places


def calculate_standard_distance(location_data:pd.DataFrame) -> float:
    """Calculate the standard distance of a set of locations.
    
    Args:
        location_data: DataFrame with latitude and longitude columns
        
    Returns:
        Standard distance of the locations
    """
    # check if there are enough locations to calculate the standard distance
    if len(location_data) <=1:
        return 0.0


    # handle antimeridian issue
    location_data = handle_antimeridian(location_data)

    # calculate the geographic mean center
    center = calculate_geographic_mean_center(location_data)


    # calculate the haversine distance from each point to the mean center
    distances = location_data.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], center['latitude'], center['longitude']),
        axis=1
    )

    # calculate the standard distance which is the square root of the average of the squared distances
    return np.sqrt(np.mean(distances**2))    

