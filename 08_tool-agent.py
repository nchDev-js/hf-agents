from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, InferenceClientModel

# Instantiate the LLM model; pass any necessary arguments (model_id, token, etc).
model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# Instantiate the agent, providing a list of tools (even if empty) and the model
agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

# Run the agent with your task
result = agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
print(result)
