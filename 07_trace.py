import os
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from smolagents import CodeAgent, InferenceClientModel

from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()  

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    raise ValueError("Langfuse API keys not set in environment variables!")

# Initialize Smolagents instrumentation
SmolagentsInstrumentor().instrument()

# Create Langfuse client
langfuse = get_client()

# Verify authentication
if langfuse.auth_check():
    print("✅ Langfuse client is authenticated and ready!")
else:
    print("❌ Authentication failed. Please check your credentials and host.")


# agent = CodeAgent(tools=[], model=InferenceClientModel())
# alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)
# alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")  
