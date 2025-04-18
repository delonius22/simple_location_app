from geopy.geocoders import Nominatim
import urllib.request
import ssl
import certifi
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def geocode_with_proxy(address, proxy_url=None, user_agent="geopy_user", timeout=10, max_retries=3):
    """
    Geocode an address using Nominatim with proxy support.
    
    Args:
        address (str): The address to geocode
        proxy_url (str): Proxy URL with authentication in format http://username:password@host:port
        user_agent (str): User agent for Nominatim requests
        timeout (int): Timeout for requests in seconds
        max_retries (int): Maximum number of retries on failure
        
    Returns:
        tuple: (latitude, longitude) or None if geocoding fails
    """
    # Configure proxy if provided
    if proxy_url:
        # Set up proxy handler
        proxy_handler = urllib.request.ProxyHandler({
            'http': proxy_url,
            'https': proxy_url
        })
        
        # Set up SSL context with proper certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        
        # Build and install opener with the proxy handler
        opener = urllib.request.build_opener(proxy_handler, https_handler)
        urllib.request.install_opener(opener)
        logger.info(f"Configured proxy: {proxy_url.replace('http://', '').split('@')[1]}")
    
    # Create the Nominatim geocoder
    # The urllib opener configuration will be used automatically
    geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
    
    # Try geocoding with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Geocoding attempt {attempt+1}/{max_retries} for: {address}")
            location = geolocator.geocode(address)
            
            if location:
                logger.info(f"Successfully geocoded: {address}")
                return (location.latitude, location.longitude)
            else:
                logger.warning(f"No results found for: {address}")
                return None
                
        except Exception as e:
            logger.error(f"Error during geocoding attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff: wait longer between retries
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries reached. Geocoding failed for: {address}")
                return None


def batch_geocode(addresses, proxy_url=None, user_agent="geopy_user", timeout=10, 
                  max_retries=3, delay=1.0):
    """
    Batch geocode multiple addresses with rate limiting.
    
    Args:
        addresses (list): List of addresses to geocode
        proxy_url (str): Proxy URL with authentication
        user_agent (str): User agent for Nominatim requests
        timeout (int): Timeout for requests in seconds
        max_retries (int): Maximum number of retries on failure
        delay (float): Delay between requests in seconds to avoid rate limiting
        
    Returns:
        dict: Dictionary of address to (latitude, longitude) mappings
    """
    results = {}
    
    for address in addresses:
        logger.info(f"Processing address: {address}")
        
        # Geocode the address
        coords = geocode_with_proxy(
            address, 
            proxy_url=proxy_url,
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Store the result
        results[address] = coords
        
        # Delay to avoid rate limiting
        if delay > 0:
            logger.info(f"Waiting {delay} seconds before next request...")
            time.sleep(delay)
    
    return results


if __name__ == "__main__":
    # Example usage
    # Replace with your actual proxy information
    proxy_url = "http://username:password@proxy_host:proxy_port"
    
    # Single address example
    address = "175 5th Avenue, New York, NY"
    result = geocode_with_proxy(address, proxy_url=proxy_url, user_agent="my_geocoding_app")
    if result:
        lat, lon = result
        print(f"Coordinates for {address}: {lat}, {lon}")
    
    # Batch geocoding example
    addresses = [
        "Empire State Building, New York, NY",
        "Golden Gate Bridge, San Francisco, CA",
        "White House, Washington, DC"
    ]
    
    # Use a longer delay for batch requests to be considerate to the Nominatim service
    results = batch_geocode(addresses, proxy_url=proxy_url, user_agent="my_geocoding_app", delay=1.5)
    
    # Print batch results
    print("\nBatch Geocoding Results:")
    for address, coords in results.items():
        if coords:
            lat, lon = coords
            print(f"{address}: {lat}, {lon}")
        else:
            print(f"{address}: Geocoding failed")