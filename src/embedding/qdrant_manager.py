# qdrant_manager.py

import logging
from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient, models



# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class QdrantManager:
    """A simple interface for managing operations in a Qdrant database."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("QdrantManager")
        
        # Read necessary settings from the config
        connection_config = config.get('connection', {})
        collection_config = config.get('collection', {})

        timeout = connection_config.get('timeout', 60.0) # Default 60 seconds

        # --- FUTURE IMPROVEMENT ---
        # TODO: Add support for Qdrant Cloud connection using settings from config.yaml.
        # This would involve checking for 'connection.cloud_url' and an API key.
        # If they exist, initialize the client with `url=...` and `api_key=...`,
        # otherwise, fall back to the local connection below. This allows switching
        # between local and cloud environments without code changes.
        
        # Focusing on local connection
        host = connection_config.get('host', 'localhost')
        port = connection_config.get('port', 6333)
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.logger.info(f"Connected to Qdrant: http://{host}:{port} (Timeout: {timeout}s)")
        
        self.collection_name = collection_config.get('name', 'jotform_help_vectors')
        
        # Vector size depends on the embedding model and should be taken from config.
        # It must match 'models.primary.dimensions' in config.yaml.
        self.vector_size = collection_config.get('vector_size', 768)

        self.batch_size = config.get('processing', {}).get('batch_size', 100)
        
        # Check if the collection exists and create it if necessary
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensures the collection exists, creating it if it doesn't."""
        try:
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"Collection '{self.collection_name}' not found, creating a new one...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info("✅ Collection created successfully.")
            else:
                self.logger.info(f"✅ Collection '{self.collection_name}' already exists.")
        except Exception as e:
            self.logger.error(f"Error during collection check/creation: {e}")
            raise

    def insert_vectors(self, chunks_data: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Uploads chunk data and their corresponding embeddings to Qdrant.
        """
        if len(chunks_data) != len(embeddings):
            self.logger.error("Number of chunks does not match the number of embeddings! Upload canceled.")
            return

        self.logger.info(f"Uploading {len(chunks_data)} vectors to Qdrant...")
        
        # Prepare the 'points' to be uploaded to Qdrant
        points_to_insert = [
            models.PointStruct(
                id=chunk['id'], # The UUID generated in our chunker
                vector=vector.tolist(),
                payload={ # Payload is all metadata other than the vector
                    "content": chunk['content'],
                    "metadata": chunk['metadata']
                }
            )
            for chunk, vector in zip(chunks_data, embeddings)
        ]
        
        # Uploading points in batches, not all at once.
        try:
            for i in range(0, len(points_to_insert), self.batch_size):
                batch = points_to_insert[i : i + self.batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                self.logger.info(f"Batch {i//self.batch_size + 1} ({len(batch)} vectors) uploaded...")
            
            self.logger.info(f"✅ Successfully uploaded {len(points_to_insert)} vectors.")
        except Exception as e:
            self.logger.error(f"Error during vector upload: {e}")
            raise
            
    def search(self, query_vector: List[float], limit: int = 5) -> List[models.ScoredPoint]:
        """
        Finds the most similar results for a given query vector.
        """
        self.logger.info(f"Searching in collection '{self.collection_name}'...")
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            self.logger.info(f"Found {len(search_results)} results.")
            return search_results
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []