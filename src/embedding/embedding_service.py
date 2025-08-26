# embedding_service.py

import json
import os
import yaml
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# qdrant_manager'ı aynı klasörden import ediyoruz
from src.embedding.qdrant_manager import QdrantManager

class SimpleEmbeddingService:
    """
    Chunk'ları okuyan, vektörlere dönüştüren ve Qdrant'a yükleyen basit servis.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        # 1. Ayarları Yükle
        print(f"📋 Ayarlar '{config_path}' dosyasından yükleniyor...")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ HATA: Config dosyası bulunamadı: {config_path}")
            raise
        
        # 2. Gerekli Parametreleri Config'den Oku (DÜZELTİLDİ)
        # Artık doğru ve iç içe geçmiş anahtarlardan okuma yapıyoruz.
        models_config = self.config.get('models', {})
        processing_config = self.config.get('processing', {})
        chunking_config = self.config.get('chunking', {})

        self.model_name = models_config.get('primary', {}).get('model_name', 'all-MiniLM-L6-v2')
        self.batch_size = processing_config.get('batch_size', 32)
        
        # Mantıksal olarak, chunker'ın çıktı dosyası, embedder'ın girdi dosyasıdır.
        self.chunks_input_file = chunking_config.get('output_file', 'data/chunks/chunks.json')
        
        # 3. Embedding Modelini Yükle
        print(f"🤖 '{self.model_name}' embedding modeli yükleniyor...")
        self.model = SentenceTransformer(self.model_name)
        print("✅ Model başarıyla yüklendi.")
        
        # 4. Qdrant Manager'ı Başlat
        # qdrant_manager kendi ayarlarını config'in 'qdrant' bölümünden alacak.
        self.qdrant_manager = QdrantManager(self.config['qdrant'])

    def load_chunks_from_file(self) -> List[Dict[str, Any]]:
        """
        Bizim sadeleştirilmiş chunker'ımızın ürettiği chunks.json dosyasını okur.
        """
        try:
            print(f"📄 '{self.chunks_input_file}' dosyasından chunk'lar okunuyor...")
            with open(self.chunks_input_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"👍 {len(chunks)} adet chunk başarıyla okundu.")
            return chunks
        except FileNotFoundError:
            print(f"❌ HATA: Chunks dosyası bulunamadı -> {self.chunks_input_file}")
            return []

    def run_pipeline(self):
        """
        Tüm embedding ve yükleme sürecini çalıştırır.
        """
        # Adım 1: Chunk verisini dosyadan yükle
        chunks = self.load_chunks_from_file()
        if not chunks:
            print("İşlem durduruldu.")
            return

        # Adım 2: Vektöre dönüştürülecek metinleri hazırla
        texts_to_embed = [
            f"Title: {chunk['metadata'].get('title', '')}\n{chunk['content']}" 
            for chunk in chunks
        ]

        # Adım 3: Metinleri toplu halde (batch) vektörlere dönüştür
        print(f"🧠 {len(texts_to_embed)} metin, {self.batch_size} boyutlu gruplar halinde vektöre dönüştürülüyor...")
        
        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        print("✅ Tüm metinler başarıyla vektöre dönüştürüldü.")

        # Adım 4: Vektörleri Qdrant'a yükle
        print("🗄️ Vektörler Qdrant veritabanına yükleniyor...")
        self.qdrant_manager.insert_vectors(
            chunks_data=chunks,
            embeddings=embeddings
        )
        print("\n🎉 Embedding ve yükleme işlemi başarıyla tamamlandı!")


def main():
    """Ana başlangıç fonksiyonu."""
    config_path = 'config/config.yaml'
    
    # Servisi başlat (tüm ayarları kendi içinde config'den okuyacak)
    service = SimpleEmbeddingService(config_path=config_path)
    
    # Tüm süreci çalıştır
    service.run_pipeline()


if __name__ == "__main__":
    main()