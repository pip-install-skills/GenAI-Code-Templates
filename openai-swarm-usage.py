from swarm import Swarm, Agent
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

import base64
import requests
import os

load_dotenv()

# Define agent functions
def get_weather(location, time="now"):
    """
    Fetch live weather data for a given location.
    
    Parameters:
    location (str): City name or coordinates (e.g., "London" or "lat,lon").
    time (str): Currently supports "now" for current weather (default).
    
    Returns:
    dict: Weather data or an error message.
    """
    API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")  # Replace with your OpenWeatherMap API key
    BASE_URL = "https://api.openweathermap.org/data/2.5/"
    
    if time == "now":
        endpoint = f"{BASE_URL}weather"
    else:
        return {"error": "Only 'now' is supported for time parameter."}
    
    params = {
        "q": location,
        "appid": API_KEY,
        "units": "metric"  # Use "imperial" for Fahrenheit
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        data = response.json()
        return {
            "location": data["name"],
            "temperature": data["main"]["temp"],
            "weather": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except KeyError:
        return {"error": "Unexpected response format. Please check the API or your query."}
    
def send_email(recipient, subject, body, attachments=None):
    message = Mail(
        from_email=os.getenv("MAIL_USERNAME"),
        to_emails=recipient,
        subject=subject,
        html_content=body)

    if attachments:
        for attachment in attachments:
            with open(attachment['file_path'], 'rb') as f:
                data = f.read()
                encoded_file = base64.b64encode(data).decode()

            attached_file = Attachment(
                FileContent(encoded_file),
                FileName(attachment['file_name']),
                FileType(attachment['file_type']),
                Disposition('attachment')
            )
            message.attachment = attached_file

    try:
        sg = SendGridAPIClient(os.getenv("MAIL_API_KEY"))
        response = sg.send(message)
        if response.status_code == 202:  # 202 indicates email was accepted by SendGrid
            return "Email sent successfully."
        else:
            print(f"Failed to send email. Status code: {response.status_code}")
            return "Email not send."
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return e

# Define agents
weather_agent = Agent(
    name="Assistant Agent",
    instructions="Help the user with their queries and requests, use the functions if needed",
    functions=[get_weather, send_email],
)
# Initialise Swarm client and run conversation
client = Swarm()
response = client.run(
    agent=weather_agent,
    messages=[{"role": "user", "content": "send an email to phanitallapudi@gmail.com with subject as 'Test' and body as how are you doing mate?"}],
)
print(response.messages[-1]["content"])
