# src/tools/rag_tool.py

from langchain.tools import Tool
from src.agents.rag_agent import RAGAgent
from dotenv import load_dotenv

dotenv_path = 'config/.env'
load_dotenv(dotenv_path=dotenv_path)

# Create a single, shared instance of the RAGAgent.
# This is efficient because the models inside RAGAgent (like SentenceTransformer)
# are loaded into memory only once when the application starts.
_rag_expert = RAGAgent()

# Now, we define the tool that our main ActionAgent can use.
rag_tool = Tool(
    name="search_knowledge_base",
    func=_rag_expert.query,
    description=(
        "Use this tool to find information and answer questions about Jotform. "
        "It is your primary source of knowledge for Jotform's features, how-to guides, "
        "tutorials, and troubleshooting. The input should be a clear, specific question."
    )
)

# This is a simple test block to demonstrate how the tool works.
if __name__ == '__main__':
    print("--- Testing the RAG Tool ---")
    
    # The ActionAgent will use the tool like this:
    question = "How can I add a payment integration to my form?"
    
    # .invoke() is the standard way to run a LangChain tool.
    result = rag_tool.invoke(question)
    
    print(f"\nQuestion asked to the tool: {question}")
    print(f"\nResult from the tool:\n{result}")
    print("\n--- Test complete ---")