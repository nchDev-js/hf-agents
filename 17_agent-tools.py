from smolagents import CodeAgent, InferenceClientModel, Tool, DuckDuckGoSearchTool
from tools.hub_stats_tool import HubStatsTool
from tools.weather_info_tool import WeatherInfoTool

# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the tool
weather_info_tool = WeatherInfoTool()

# Initialize the tool
hub_stats_tool = HubStatsTool()

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[search_tool, weather_info_tool, hub_stats_tool], 
    model=model
)

# Example usage
print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook

# Example query Alfred might receive during the gala
response = alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)