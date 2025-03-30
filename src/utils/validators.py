

import uuid
import ipaddress
import datetime
from typing import Dict, List, Any, Iterator, Tuple, Optional, Union
from itertools import islice


def format_timestamp(timestamp: Union[datetime.datetime, str], 
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a timestamp to a consistent string format.
    
    Args:
        timestamp: Datetime object or ISO format timestamp string
        format_str: Output format string
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return timestamp.strftime(format_str)


def calculate_date_ranges(
    days: int = 30, 
    end_date: Optional[datetime.datetime] = None
) -> Tuple[datetime.datetime, datetime.datetime]:
    """Calculate start and end dates for a given time range.
    
    Args:
        days: Number of days to include in the range
        end_date: Optional end date (defaults to now)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if end_date is None:
        end_date = datetime.datetime.now(datetime.timezone.utc)
    
    start_date = end_date - datetime.timedelta(days=days)
    
    return start_date, end_date


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (values override dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def chunked_iterable(iterable: Iterator[Any], size: int) -> Iterator[List[Any]]:
    """Split an iterable into chunks of a specified size.
    
    Args:
        iterable: The iterable to chunk
        size: Maximum chunk size
        
    Returns:
        Iterator that yields chunks of the iterable
    """
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


def anonymize_ip(ip_address: str) -> str:
    """Anonymize an IP address by zeroing out the last part.
    
    Args:
        ip_address: Original IP address
        
    Returns:
        Anonymized IP address
    """
    try:
        if ':' in ip_address:  # IPv6
            ip = ipaddress.IPv6Address(ip_address)
            # Mask the last 80 bits (keep first 48 bits)
            network = ipaddress.IPv6Network(f"{ip}/48", strict=False)
            return str(network.network_address)
        else:  # IPv4
            ip = ipaddress.IPv4Address(ip_address)
            # Mask the last octet (keep first 24 bits)
            network = ipaddress.IPv4Network(f"{ip}/24", strict=False)
            return str(network.network_address)
    except ValueError:
        # Return original if invalid
        return ip_address


def is_valid_coordinates(latitude: float, longitude: float) -> bool:
    """Check if coordinates are valid.
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def generate_unique_id() -> str:
    """Generate a unique ID for tracking purposes.
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())