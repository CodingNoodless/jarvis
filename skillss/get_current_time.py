from langchain.tools import BaseTool
from datetime import datetime
import pytz
from typing import Optional, Type
from pydantic import BaseModel, Field

class TimeInput(BaseModel):
    """Input for the time tool"""
    timezone: str = Field(
        default="local",
        description="The timezone to get the time for. Use 'local' for local time, or timezone names like 'UTC', 'America/New_York', 'Europe/London', etc."
    )

class TimeSkill(BaseTool):
    """Tool for getting current time in various timezones"""
    
    name: str = "get_current_time"
    description: str = (
        "Get the current time in a specified timezone. "
        "Input MUST be a valid timezone name (e.g. 'America/Los_Angeles', 'UTC', 'Europe/London') "
        "or a supported city name (e.g. 'San Francisco', 'London', 'Tokyo'). "
        "EXAMPLES: Action Input: San Francisco, Action Input: America/Los_Angeles. "
        "Do NOT provide explanations, offsets, or extra text—just the timezone or city name."
    )
    args_schema: Type[BaseModel] = TimeInput
    
    def _run(self, timezone: str = "local") -> str:
        """Get current time in specified timezone"""
        try:
            # Clean up the timezone input
            timezone = timezone.strip().strip("'\"")
            
            # Handle common variations and city names
            if timezone.lower() in ["local", "local time", ""]:
                current_time = datetime.now()
                return f"Current local time is {current_time.strftime('%I:%M %p on %A, %B %d, %Y')}"
            
            # Handle common timezone aliases and city names
            timezone_map = {
                "utc": "UTC",
                "gmt": "GMT",
                "est": "America/New_York",
                "eastern": "America/New_York",
                "pst": "America/Los_Angeles",
                "pacific": "America/Los_Angeles",
                "cst": "America/Chicago",
                "central": "America/Chicago",
                "mst": "America/Denver",
                "mountain": "America/Denver",
                "ist": "Asia/Kolkata",
                "india": "Asia/Kolkata",
                "jst": "Asia/Tokyo",
                "japan": "Asia/Tokyo",
                "bst": "Europe/London",
                "london": "Europe/London",
                "new york": "America/New_York",
                "los angeles": "America/Los_Angeles",
                "chicago": "America/Chicago",
                "denver": "America/Denver",
                "tokyo": "Asia/Tokyo",
                "mumbai": "Asia/Kolkata",
                "delhi": "Asia/Kolkata",
                "paris": "Europe/Paris",
                "berlin": "Europe/Berlin",
                "sydney": "Australia/Sydney",
                "melbourne": "Australia/Melbourne",
                "san francisco": "America/Los_Angeles"
            }
            
            # Use mapped timezone if available, otherwise use as-is
            actual_timezone = timezone_map.get(timezone.lower(), timezone)
            
            # Get timezone object
            tz = pytz.timezone(actual_timezone)
            current_time = datetime.now(tz)
            
            # Format the response nicely
            time_str = current_time.strftime('%I:%M %p on %A, %B %d, %Y')
            return f"Current time in {timezone} is {time_str} ({actual_timezone})"
                
        except pytz.exceptions.UnknownTimeZoneError:
            # Provide helpful suggestions
            suggestions = [
                "UTC", "America/New_York", "America/Los_Angeles", "Europe/London", 
                "Asia/Tokyo", "Australia/Sydney", "local"
            ]
            return (
                f"Unknown timezone: '{timezone}'. "
                f"Please use a valid timezone name. Some examples: {', '.join(suggestions)}"
            )
        except Exception as e:
            return f"Error getting time: {str(e)}"
    
    async def _arun(self, timezone: str = "local") -> str:
        """Async version of _run"""
        return self._run(timezone)

# Create the skill instance that the SkillManager will look for
skill = TimeSkill()