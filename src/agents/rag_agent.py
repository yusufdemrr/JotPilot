# src/agents/rag_agent.py

import yaml
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

# We need to import our existing, powerful modules
from src.llm.openai_client import OpenAIClient
from src.embedding.qdrant_manager import QdrantManager

class RAGAgent:
    """
    A specialized agent that answers questions based on a Qdrant knowledge base.
    It encapsulates the entire RAG pipeline: embedding a query, searching for context,
    and generating a response with an LLM. It acts as a "knowledge expert".
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initializes the RAGAgent and all its necessary components.
        
        Args:
            config_path (str): Path to the main configuration file.
        """
        print("🤖 Initializing RAGAgent (Knowledge Expert)...")
        
        # Load the central configuration file
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize the necessary components by passing them their specific config sections
        self.openai_client = OpenAIClient(self.config['llm'])
        self.qdrant_manager = QdrantManager(self.config['qdrant'])
        
        # This agent needs its own embedding model to convert user questions into vectors.
        # It MUST be the same model used to embed the documents in the first place.
        model_name = self.config['models']['primary']['model_name'] #? embedding model name
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load the prompt templates from the file paths specified in the config
        print("   - Loading prompt templates from files...")
        try:
            # Get the file paths from the config
            system_prompt_path = self.config['llm']['prompts']['rag_system_prompt_path']
            rag_template_path = self.config['llm']['prompts']['rag_template_path']

            # Read the content from the system prompt file
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()

            # Read the content from the rag template file
            with open(rag_template_path, 'r', encoding='utf-8') as f:
                self.rag_template = f.read()
            
            print("   - Prompts loaded successfully.")

        except FileNotFoundError as e:
            print(f"❌ ERROR: Prompt file not found: {e}")
            raise
        except KeyError as e:
            print(f"❌ ERROR: Prompt path key not found in config.yaml: {e}")
            raise

        print("✅ RAGAgent initialized successfully.")

    def query(self, question: str) -> str:
        """
        Takes a user's question, processes it through the RAG pipeline, 
        and returns a string answer. This is the main method of the agent.

        Args:
            question (str): The question to answer.

        Returns:
            str: The generated answer from the LLM based on the retrieved context.
        """
        print(f"🧠 RAGAgent received a question: '{question}'")
        
        # Step 1: Embed the user's question into a vector
        print("   - Step 1: Embedding the question...")
        query_vector = self.embedding_model.encode(question).tolist()
        
        # Step 2: Retrieve relevant context from Qdrant
        print("   - Step 2: Searching for context in Qdrant...")
        search_limit = self.config['llm']['rag'].get('search_limit', 5)
        search_results = self.qdrant_manager.search(
            query_vector=query_vector,
            limit=search_limit
        )
        
        if not search_results:
            print("   - No context found in the knowledge base.")
            return "I'm sorry, but I couldn't find any information on that topic in my knowledge base."
            
        # Step 3: Augment the prompt with the retrieved context
        print(f"   - Step 3: Found {len(search_results)} relevant documents. Augmenting prompt...")
        context = "\n\n---\n\n".join([
            result.payload['content'] for result in search_results
        ])
        
        final_prompt = self.rag_template.format(
            context=context,
            question=question
        )
        
        # Step 4: Generate the final answer using the LLM
        print("   - Step 4: Generating final answer with LLM...")
        final_answer = self.openai_client.get_completion(
            system_prompt=self.system_prompt,
            user_prompt=final_prompt
        )
        
        return final_answer

# This block is for simple, direct testing of this file.
if __name__ == '__main__':
    # This demonstrates how another part of our system can use this agent.
    rag_expert = RAGAgent()
    
    test_question = "How do I create a new form?"
    answer = rag_expert.query(test_question)
    
    print("\n--- RAG AGENT TEST ---")
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
    print("----------------------")