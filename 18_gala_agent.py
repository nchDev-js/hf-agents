# Import necessary libraries
import datasets
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
from langchain_core.documents import Document

# Import our custom tools from their modules
from guest_info_retriever_tool import GuestInfoRetrieverTool
from tools.hub_stats_tool import HubStatsTool
from tools.weather_info_tool import WeatherInfoTool

# Initialize the Hugging Face model
model = InferenceClientModel()

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
guest_list = list(guest_dataset)

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_list
]

# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)

query = "Tell me about 'Lady Ada Lovelace'"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)


query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)

query = "One of our guests is from Qwen. What can you tell me about their most popular model?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)

query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)


# Create Alfred with conversation memory
alfred_with_memory = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,
    planning_interval=3
)

# First interaction
response1 = alfred_with_memory.run("Tell me about Lady Ada Lovelace.")
print("🎩 Alfred's First Response:")
print(response1)

# Second interaction (referencing the first)
response2 = alfred_with_memory.run("What projects is she currently working on?", reset=False)
print("🎩 Alfred's Second Response:")
print(response2)