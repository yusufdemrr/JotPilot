# intelligent_chunker.py

import json
import argparse
import os
import re
import uuid
from typing import List, Dict, Any
import tiktoken
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SimpleChunker:
    """
    Crawler'dan gelen metni RAG için temel parçalara ayıran basit sistem.
    - Karmaşık Parent Document, içerik tespiti ve dinamik overlap kaldırıldı.
    - Crawler'ın yeni çıktısıyla (--- PAGE BREAK ---) uyumlu hale getirildi.
    """
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo"):
        """
        Chunker'ı temel parametrelerle başlatır.
        
        Args:
            chunk_size (int): Her bir parçanın hedef token boyutu.
            chunk_overlap (int): Parçalar arasındaki örtüşme (token cinsinden).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Token sayacı (encoder)
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
            print(f"✅ Model için token sayacı yüklendi: {model_name}")
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            print(f"⚠️ Uyarı: Model bulunamadı, varsayılan encoder kullanılıyor.")
            
        print(f"✅ SimpleChunker başlatıldı. Hedef boyut: {chunk_size} token, Örtüşme: {chunk_overlap} token.")

    def count_tokens(self, text: str) -> int:
        """Verilen metnin token sayısını döndürür."""
        return len(self.encoder.encode(text))

    def parse_pages_from_txt(self, txt_file_path: str) -> List[Dict[str, str]]:
        """
        Sadeleştirilmiş crawler'ın YENİ çıktısını (URL ve TITLE içeren) okur.
        """
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"❌ Hata: Giriş dosyası bulunamadı -> {txt_file_path}")
            return []

        pages = []
        raw_pages = content.split('--- PAGE BREAK ---')
        
        for raw_page in raw_pages:
            raw_page = raw_page.strip()
            if not raw_page:
                continue
            
            lines = raw_page.split('\n')
            if len(lines) < 2:  # En azından URL, TITLE ve içerik olmalı
                continue
            
            url_line = lines[0]
            title_line = lines[1]
            
            url_match = re.match(r"URL: (https?://[^\s]+)", url_line)
            title_match = re.match(r"TITLE: (.+)", title_line)

            if url_match and title_match:
                url = url_match.group(1).strip()
                title = title_match.group(1).strip()
                # Geriye kalan her şey içeriktir
                page_content = '\n'.join(lines[2:]).strip()
                pages.append({"url": url, "title": title, "content": page_content})
            
        
        print(f"📄 {len(pages)} adet sayfa (başlık bilgisiyle) başarıyla okundu.")
        return pages

    def create_chunks(self, pages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Okunan sayfalardan parçacıklar (chunk) oluşturur.
        """
        all_chunks = []
        
        # LangChain'in hazır ve etkili text splitter'ını kullanalım.
        # Bu, kendi 'split_text_smartly' fonksiyonumuzu yazmaktan daha basit ve standart bir yoldur.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", " ", ""] # Bölme öncelik sırası
        )
        
        for page in pages:
            # Metni LangChain splitter ile böl
            chunks_content = text_splitter.split_text(page['content'])
            
            for i, chunk_text in enumerate(chunks_content):
                chunk = {
                    "id": str(uuid.uuid4()), # Her chunk için eşsiz bir ID
                    "content": chunk_text,
                    "metadata": {
                        "source_url": page['url'],
                        "title": page['title'],
                        "chunk_index": i + 1,
                        "token_count": self.count_tokens(chunk_text)
                    }
                }
                all_chunks.append(chunk)
        
        print(f"🎉 Toplam {len(all_chunks)} adet chunk oluşturuldu.")
        return all_chunks

    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_file: str):
        """Oluşturulan chunk'ları basit bir JSON dosyasına kaydeder."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Chunk'lar başarıyla '{output_file}' dosyasına kaydedildi.")

def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """YAML config dosyasını yükler."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                print(f"📋 Ayarlar şuradan yükleniyor: {config_path}")
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"❌ Config dosyası okunurken hata oluştu: {e}")
    return {}

def main():
    parser = argparse.ArgumentParser(description="Metin dosyasını RAG için parçalara (chunk) böler.")
    parser.add_argument("input_file", help="Crawler tarafından oluşturulan girdi .txt dosyası.")
    parser.add_argument("--output", "-o", default="data/chunks/chunks.json", help="Çıktı JSON dosyasının yolu.")
    parser.add_argument("--chunk-size", "-s", type=int, default=400, help="Hedef chunk boyutu (token).")
    parser.add_argument("--chunk-overlap", "-v", type=int, default=50, help="Chunk'lar arası örtüşme (token).")

    args = parser.parse_args()

    # --- CONFIG ---
    config = load_config()
    chunking_config = config.get('chunking', {})

    # Değerleri belirle (Komut satırı > config dosyası > varsayılan)
    input_file = args.input_file or chunking_config.get('input_file', 'data/raw/jotform_trial.txt')
    output_file = args.output or chunking_config.get('output_file', 'data/chunks/chunks.json')
    chunk_size = args.chunk_size or chunking_config.get('target_chunk_size', 400)
    chunk_overlap = args.chunk_overlap or chunking_config.get('overlap_size', 50)
    # --- CONFIG ---


    # Chunker'ı başlat
    chunker = SimpleChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Crawler çıktısını oku ve sayfalara ayır
    pages = chunker.parse_pages_from_txt(input_file)

    if pages:
        chunks = chunker.create_chunks(pages)
        if chunks:
            chunker.save_chunks_to_json(chunks, output_file)


if __name__ == "__main__":
    main()