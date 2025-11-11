import os
from smolagents import CodeAgent, InferenceClientModel, tool
from dotenv import load_dotenv
load_dotenv()  # loads .env file

HF_TOKEN = os.environ.get("HF_TOKEN")

@tool
def get_weather_data(city: str) -> dict:
    """
    Returns sample weather data for a given city

    Args:
        city: Name of the city (new york, london or tokyo)
    """

    sample_data = {
        "new york": {
            "temps": [72, 75, 65, 68, 70, 74, 73],
            "rain": [0, 0.2, 0.5, 0, 0, 0.1, 0],
            "unit": "F"
        },
        "london": {
            "temps": [15, 16, 17, 14, 18, 19, 15],  # Celsius
            "rain": [0.3, 0.1, 0.6, 0.2, 0.4, 0.3, 0],
            "unit": "C"
        },
        "tokyo": {
            "temps": [25, 26, 27, 28, 27, 26, 25],  # Celsius
            "rain": [0, 0.1, 0.4, 0.3, 0, 0.2, 0],
            "unit": "C"
        }
    }

    city_lower = city.lower()
    return sample_data.get(city_lower, {"error": f"No data for {city}"})


agent = CodeAgent(
    tools=[get_weather_data],
    model=InferenceClientModel(),
    additional_authorized_imports=["matplotlib"],
    verbosity_level=2
)

agent.push_to_hub(
    repo_id="WeatherAgent",               
    commit_message="my first shared agent",
    private=False,
    token=os.getenv("HF_TOKEN_WRITE"),
)
