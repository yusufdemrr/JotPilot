# embedding_service.py (NİHAİ VE ESNEK VERSİYON)

import json
import os
import yaml
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.embedding.qdrant_manager import QdrantManager

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class SimpleEmbeddingService:
    """
    Reads chunks, converts them to vectors using the provider specified in the config
    (HuggingFace or OpenAI), and uploads them to Qdrant.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        print(f"📋 Loading settings from '{config_path}'...")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Config file not found: {config_path}")
            raise
        
        # --- Read settings from config ---
        models_config = self.config.get('models', {})
        processing_config = self.config.get('processing', {})
        chunking_config = self.config.get('chunking', {})
        
        # Determine the primary provider (e.g., "huggingface" or "openai")
        primary_provider_config = models_config.get('primary', {})
        self.provider = primary_provider_config.get('provider', 'huggingface')
        self.model_name = primary_provider_config.get('model_name', 'all-MiniLM-L6-v2')
        
        self.batch_size = processing_config.get('batch_size', 32)
        self.chunks_input_file = chunking_config.get('output_file', 'data/chunks/chunks.json')
        
        # --- Initialize client/model based on the provider ---
        self.model = None
        self.openai_client = None

        if self.provider == 'huggingface':
            print(f"🤖 Initializing HuggingFace model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        elif self.provider == 'openai':
            if OpenAI is None:
                raise RuntimeError("OpenAI package is not installed. Please run `pip install openai`.")
            
            # Find the correct API key env variable name
            api_key_env_name = primary_provider_config.get('api_key_env') or \
                               models_config.get('fallback', {}).get('api_key_env') or \
                               "OPENAI_API_KEY"
            
            api_key = os.getenv(api_key_env_name)
            if not api_key:
                raise ValueError(f"OpenAI API key not found in environment variable '{api_key_env_name}'.")
            
            print(f"🤖 Initializing OpenAI client for model: {self.model_name}")
            self.openai_client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider in config.yaml: '{self.provider}'. Must be 'huggingface' or 'openai'.")
        
        self.qdrant_manager = QdrantManager(self.config['qdrant'])
        print("✅ EmbeddingService initialized successfully.")

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Private helper method to create embeddings based on the configured provider.
        This provides a single interface for different embedding methods.
        """
        if self.provider == 'huggingface':
            print(f"   - Generating embeddings with HuggingFace model...")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True
            )
            return np.asarray(embeddings, dtype=np.float32)
        
        elif self.provider == 'openai':
            print(f"   - Generating embeddings with OpenAI model...")
            all_embeddings = []
            # Use tqdm for progress bar
            for i in tqdm(range(0, len(texts), self.batch_size), desc="OpenAI Embedding"):
                batch = texts[i : i + self.batch_size]
                try:
                    response = self.openai_client.embeddings.create(model=self.model_name, input=batch)
                    all_embeddings.extend([d.embedding for d in response.data])
                except Exception as e:
                    print(f"❌ ERROR during OpenAI batch processing: {e}")
                    # Add empty vectors for the failed batch to maintain list size,
                    # or handle more gracefully.
                    all_embeddings.extend([np.zeros(self.config['models']['primary']['dimensions']).tolist()] * len(batch))

            return np.asarray(all_embeddings, dtype=np.float32)

    def load_chunks_from_file(self) -> List[Dict[str, Any]]:
        try:
            print(f"📄 Loading chunks from '{self.chunks_input_file}'...")
            with open(self.chunks_input_file, 'r', encoding='utf-8') as f:
                # Assuming the file contains a list of chunks directly at the top level
                data = json.load(f)
                if isinstance(data, dict) and 'chunks' in data:
                    chunks = data['chunks']
                elif isinstance(data, list):
                    chunks = data
                else:
                    raise ValueError("Invalid chunks file format. Expected a list or a dict with a 'chunks' key.")
            print(f"👍 Successfully loaded {len(chunks)} chunks.")
            return chunks
        except FileNotFoundError:
            print(f"❌ ERROR: Chunks file not found -> {self.chunks_input_file}")
            return []

    def run_pipeline(self):
        """
        Runs the complete embedding and upload pipeline.
        """
        # Step 1: Load chunk data from the file.
        chunks = self.load_chunks_from_file()
        if not chunks:
            print("Halting pipeline as no chunks were loaded.")
            return

        # Step 2: Prepare the texts for embedding.
        texts_to_embed = [
            f"Title: {chunk['metadata'].get('title', '')}\n{chunk['content']}" 
            for chunk in chunks
        ]

        # Step 3: Create embeddings using the correct provider.
        print(f"🧠 Creating embeddings for {len(texts_to_embed)} texts using '{self.provider}' provider...")
        embeddings = self._create_embeddings(texts_to_embed)
        print("✅ All texts successfully converted to vectors.")

        # Step 4: Upload the vectors to Qdrant.
        print("🗄️ Uploading vectors to Qdrant...")
        self.qdrant_manager.insert_vectors(
            chunks_data=chunks,
            embeddings=embeddings
        )
        print("\n🎉 Embedding and upload pipeline completed successfully!")

def main():
    config_path = 'config/config.yaml'
    # Load .env file to make environment variables available
    load_dotenv('config/.env')
    
    service = SimpleEmbeddingService(config_path=config_path)
    service.run_pipeline()

if __name__ == "__main__":
    main()