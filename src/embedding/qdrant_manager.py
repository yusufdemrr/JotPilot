# qdrant_manager.py

import logging
from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient, models

# Temel loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class QdrantManager:
    """Qdrant veritabanı işlemlerini yöneten basit arayüz."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("QdrantManager")
        
        # Gerekli ayarları config'den oku
        connection_config = config.get('connection', {})
        collection_config = config.get('collection', {})

        timeout = connection_config.get('timeout', 60.0) # Varsayılan 60 saniye
        
        # Sadece yerel bağlantıya odaklanıyoruz
        host = connection_config.get('host', 'localhost')
        port = connection_config.get('port', 6333)
        
        # Qdrant client'ı başlat
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.logger.info(f"Qdrant'a bağlanıldı: http://{host}:{port} (Timeout: {timeout}s)")
        
        self.collection_name = collection_config.get('name', 'jotform_help_vectors')
        
        # Vektör boyutu (embedding modeline göre değişir, config'den alınmalı)
        # config.yaml'daki 'models.primary.dimensions' ile eşleşmelidir.
        self.vector_size = collection_config.get('vector_size', 768)

        self.batch_size = config.get('processing', {}).get('batch_size', 100)
        
        # Koleksiyonun var olup olmadığını kontrol et ve gerekirse oluştur
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Koleksiyonun var olduğundan emin olur, yoksa oluşturur."""
        try:
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"'{self.collection_name}' koleksiyonu bulunamadı, yeni bir tane oluşturuluyor...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info("✅ Koleksiyon başarıyla oluşturuldu.")
            else:
                self.logger.info(f"✅ '{self.collection_name}' koleksiyonu zaten mevcut.")
        except Exception as e:
            self.logger.error(f"Koleksiyon kontrolü/oluşturma sırasında hata: {e}")
            raise

    def insert_vectors(self, chunks_data: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Chunk verilerini ve embedding'lerini Qdrant'a yükler.
        """
        if len(chunks_data) != len(embeddings):
            self.logger.error("Chunk sayısı ile embedding sayısı eşleşmiyor! Yükleme iptal edildi.")
            return

        self.logger.info(f"{len(chunks_data)} adet vektör Qdrant'a yükleniyor...")
        
        # Qdrant'a yüklenecek "noktaları" (points) hazırla
        points_to_insert = [
            models.PointStruct(
                id=chunk['id'], # Bizim chunker'da ürettiğimiz UUID
                vector=vector.tolist(),
                payload={ # Payload, vektör dışındaki tüm meta verilerdir
                    "content": chunk['content'],
                    "metadata": chunk['metadata']
                }
            )
            for chunk, vector in zip(chunks_data, embeddings)
        ]
        
        # Tüm noktaları tek seferde değil, batch_size'a göre bölerek yüklüyoruz.
        try:
            for i in range(0, len(points_to_insert), self.batch_size):
                batch = points_to_insert[i : i + self.batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                self.logger.info(f"Batch {i//self.batch_size + 1} ({len(batch)} vektör) yüklendi...")
            
            self.logger.info(f"✅ {len(points_to_insert)} vektör başarıyla yüklendi.")
        except Exception as e:
            self.logger.error(f"Vektör yüklemesi sırasında hata: {e}")
            raise
            
    def search(self, query_vector: List[float], limit: int = 5) -> List[models.ScoredPoint]:
        """
        Verilen bir sorgu vektörüne en benzer sonuçları bulur.
        """
        self.logger.info(f"'{self.collection_name}' içinde arama yapılıyor...")
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            self.logger.info(f"{len(search_results)} adet sonuç bulundu.")
            return search_results
        except Exception as e:
            self.logger.error(f"Arama sırasında hata: {e}")
            return []