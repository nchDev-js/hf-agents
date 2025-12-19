import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool
from langchain_core.documents import Document
import datasets
from langchain_core.documents import Document
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever

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

class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = (
        "Retrieves detailed information about gala guests and suggests "
        "conversation starters based on their background and interests."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()  # 🔴 THIS IS REQUIRED
        self.retriever = BM25Retriever.from_documents(docs)

    def _generate_conversation_starters(self, description: str) -> list[str]:
        starters = []

        if re.search(r"science|math|engineer|inventor", description, re.I):
            starters.append(
                "Ask about what first inspired their interest in science or mathematics."
            )

        if re.search(r"art|music|literature|poetry", description, re.I):
            starters.append(
                "Mention a recent exhibition or book and ask for their thoughts."
            )

        if re.search(r"politics|activist|reform|social", description, re.I):
            starters.append(
                "Ask about a social issue they care deeply about."
            )

        if not starters:
            starters.append(
                "Ask what they are most excited about attending at tonight’s gala."
            )

        return starters[:3]

    def forward(self, query: str) -> str:
        results = self.retriever.invoke(query)

        if not results:
            return "No matching guest information found."

        responses = []

        for doc in results[:3]:
            description_match = re.search(
                r"Description:\s*(.*)", doc.page_content, re.S
            )
            description = description_match.group(1) if description_match else ""

            starters = self._generate_conversation_starters(description)

            responses.append(
                f"{doc.page_content}\n\n"
                f"🗣 Conversation Starters:\n"
                + "\n".join(f"- {s}" for s in starters)
            )

        return "\n\n" + ("─" * 40 + "\n\n").join(responses)

# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)

from smolagents import CodeAgent, InferenceClientModel

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# Example query Alfred might receive during the gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)
