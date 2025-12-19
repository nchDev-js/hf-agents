import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, InferenceClientModel, Tool
from guest_info_retriever_tool import GuestInfoRetrieverTool

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

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# Example query Alfred might receive during the gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)
