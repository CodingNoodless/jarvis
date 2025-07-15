from langchain_core.tools import tool
from samsungtvws import SamsungTVWS

# Replace with your TV's IP address and token file path
tv_ip = '192.168.86.44'  # CHANGE THIS TO YOUR TV'S IP ADDRESS
token_file = 'tv-token.txt'

@tool
def list_tv_apps() -> str:
    """List the installed apps on the Samsung Smart TV."""
    tv = SamsungTVWS(host=tv_ip, port=8002, token_file=token_file)
    try:
        apps = tv.app_list()
        app_list_str = ", ".join([f"{app['name']} (ID: {app['appId']})" for app in apps])
        return app_list_str
    except Exception as e:
        return f"Error listing apps: {str(e)}"

@tool
def open_tv_app(app_name: str) -> str:
    """Open a specific app on the Samsung Smart TV by providing the app name or ID."""
    tv = SamsungTVWS(host=tv_ip, port=8002, token_file=token_file)
    try:
        apps = tv.app_list()
        app_to_open = next((app for app in apps if app['name'].lower() == app_name.lower() or app['appId'] == app_name), None)
        if app_to_open:
            tv.open_app(app_to_open['appId'])
            return f"Opened app: {app_to_open['name']}"
        else:
            return f"App '{app_name}' not found."
    except Exception as e:
        return f"Error opening app: {str(e)}"

@tool
def send_tv_key(key_code: str) -> str:
    """Send a remote control key to the Samsung Smart TV (e.g., 'KEY_VOLUP', 'KEY_POWER')."""
    tv = SamsungTVWS(host=tv_ip, port=8002, token_file=token_file)
    try:
        tv.send_key(key_code)
        return f"Sent key: {key_code}"
    except Exception as e:
        return f"Error sending key: {str(e)}"

@tool
def toggle_tv_power() -> str:
    """Toggle the power of the Samsung Smart TV.
    
    Args:
        None
    
    Returns:
        str: A message indicating the power state change or an error message.
    """
    tv = SamsungTVWS(host=tv_ip, port=8002, token_file=token_file)
    print("Toggling TV power...")
    try:
        tv.shortcuts().power()
        return "Toggled TV power"
    except Exception as e:
        return f"Error toggling power: {str(e)}"
    
skill = toggle_tv_power