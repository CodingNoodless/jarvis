from langchain_core.tools import tool
import datetime
import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

@tool
def get_current_time(location: str) -> str:
    """
    Check the current time in a specified location.

    Args:
        location (str): The location to check the current time in. Can be a city name or a timezone name.

    Returns:
        str: The current time in the specified location, or an error message if the location cannot be determined.
    """ 
    try:
        # Check if location is a valid timezone
        try:
            tz = pytz.timezone(location)
        except pytz.UnknownTimeZoneError:
            # Not a valid timezone, try to geocode
            geolocator = Nominatim(user_agent="jarvis_time_api")
            location_data = geolocator.geocode(location)
            if location_data is None:
                return f"Location '{location}' not found."
            lat, lon = location_data.latitude, location_data.longitude
            tf = TimezoneFinder()
            tz_name = tf.timezone_at(lng=lon, lat=lat)
            if tz_name is None:
                return f"Timezone not found for location '{location}'."
            tz = pytz.timezone(tz_name)
        
        # Get current time in the timezone
        current_time = datetime.datetime.now(tz)
        formatted_time = current_time.strftime("%I:%M %p")
        abbreviation = current_time.tzname()
        print("current time tool used!!!")
        return f"The current time in {location} is {formatted_time} {abbreviation}."
    
    except Exception as e:
        return f"Error: {str(e)}"

skill = get_current_time