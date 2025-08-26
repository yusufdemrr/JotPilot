# crawler.py

import asyncio
import os
import argparse
import re
import yaml
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

def clean_text_content(text: str) -> str:
    """
    Metin içeriğini RAG için temizler.
    - Gereksiz boşlukları ve satır sonlarını tek boşluğa indirir.
    - Tekrar eden veya anlamsız kısa cümleleri kaldırır.
    """
    # 1. Adım: Tüm çoklu boşlukları, tab'ları ve yeni satırları tek boşluğa indirge
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Adım: Metni cümlelere böl
    # Nokta, soru işareti, ünlemden sonra boşluk olan yerlerden böler
    sentences = re.split(r'(?<=[.?!])\s+', cleaned_text)
    
    unique_sentences = []
    seen_phrases = set() # Görülen cümle başlıklarını saklamak için
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Çok kısa veya anlamsız cümleleri atla (genellikle menü öğeleri)
        if len(sentence.split()) < 4:
            continue
            
        # Cümlenin ilk 50 karakterini (harf ve rakamlar) anahtar olarak kullan
        # Bu, "Click here for more." ve "Click here for more information." gibi
        # çok benzer cümleleri yakalamaya yardımcı olur.
        sentence_key = re.sub(r'[^\w\s]', '', sentence.lower())[:50]
        
        if sentence_key not in seen_phrases:
            seen_phrases.add(sentence_key)
            unique_sentences.append(sentence)
            
    # Temizlenmiş ve eşsiz cümleleri yeniden birleştir
    return ' '.join(unique_sentences)

def simple_extract_content(html_content: str) -> str:
    """
    HTML'den ana içeriği basitçe çıkaran ve temizleyen fonksiyon.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Adım: Sayfa başlığını (<h1>) bul
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "Başlık Bulunamadı"
    
    selectors = ['main', 'article', '[role="main"]', '.content', '#content']
    main_content_tag = None
    for selector in selectors:
        main_content_tag = soup.select_one(selector)
        if main_content_tag:
            break
            
    if not main_content_tag:
        main_content_tag = soup.body
        if not main_content_tag:
            return {"title": title, "content": ""}
        
    # h1 etiketini içerikten çıkaralım ki tekrar etmesin
    if title_tag and main_content_tag.find('h1'):
        title_tag.decompose()

    for tag in main_content_tag.select('nav, footer, script, style, aside, header'):
        tag.decompose()
        
    raw_text = main_content_tag.get_text(separator=' ', strip=True)
    
    # Ham metni, tekrarları ve boşlukları temizleyen fonksiyondan geçir
    cleaned_text = clean_text_content(raw_text)
    
    return {"title": title, "content": cleaned_text}

async def crawl_site(start_url: str, max_depth: int, max_links: int):
    """
    Belirtilen sitede gezinir, içeriği toplar ve yapılandırılmış metin döndürür.
    """
    print(f"Crawling başlıyor. Başlangıç URL: {start_url}, Derinlik: {max_depth}")
    
    all_pages_content = []
    urls_to_visit = {start_url}
    visited_urls = set()
    
    async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
        for depth in range(max_depth + 1):
            if not urls_to_visit:
                break
                
            current_level_urls = list(urls_to_visit)[:max_links]
            urls_to_visit.clear()
            
            print(f"\n--- Seviye {depth} ---")
            print(f"Gezilecek {len(current_level_urls)} URL var.")
            
            for url in current_level_urls:
                if url in visited_urls:
                    continue
                
                print(f"-> Geziliyor: {url}")
                visited_urls.add(url)
                
                try:
                    result = await crawler.arun(url=url, wait_for="networkidle")
                    
                    if not result.success or not result.html:
                        print(f"  ❌ Başarısız: {result.error_message}")
                        continue

                    page_data = simple_extract_content(result.html)
                    content = page_data['content']
                    title = page_data['title']
                    
                    if len(content) < 200:
                        print(f"  ⚠️ İçerik temizlendikten sonra çok kısa kaldı ({len(content)} karakter), atlanıyor.")
                        continue
                        
                    formatted_content = f"URL: {url}\nTITLE: {title}\n\n{content}"
                    
                    all_pages_content.append(formatted_content)
                    print(f"  ✅ Başarılı: '{title}' başlıklı sayfa eklendi.")

                    
                    if depth < max_depth:
                        soup = BeautifulSoup(result.html, 'html.parser')
                        base_domain = urlparse(url).netloc
                        for a_tag in soup.find_all('a', href=True):
                            href = a_tag['href']
                            full_url = urljoin(url, href)
                            if urlparse(full_url).netloc == base_domain and '/help/' in full_url:
                                urls_to_visit.add(full_url)
                                
                except Exception as e:
                    print(f"  ❌ Hata oluştu: {e}")

    return all_pages_content

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """YAML config dosyasını yükler ve crawling ayarlarını döndürür."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                # Sadece 'crawling' bölümünü döndür, yoksa boş dict döndür
                return config.get('crawling', {})
            except yaml.YAMLError as e:
                print(f"❌ Config dosyası okunurken hata oluştu: {e}")
    return {}

async def main(base_url: str, max_depth: int, output_file: str, max_links: int):
    """Ana fonksiyon. Crawl işlemini başlatır ve sonucu dosyaya yazar."""
    
    # base_url artık parametre olarak geliyor.
    all_content = await crawl_site(base_url, max_depth, max_links)
    
    if not all_content:
        print("\nHiç içerik bulunamadı. İşlem sonlandırılıyor.")
        return

    final_text = "\n\n--- PAGE BREAK ---\n\n".join(all_content)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    print(f"\n🎉 Crawling tamamlandı!")
    print(f"📁 Sonuçlar '{output_file}' dosyasına kaydedildi.")
    print(f"📄 Toplam {len(all_content)} sayfa işlendi.")

if __name__ == "__main__":
    # Bu blok tamamen güncellenmiştir
    parser = argparse.ArgumentParser(description="Jotform Help sayfasını gezer ve içeriği bir dosyaya kaydeder.")
    parser.add_argument("--depth", "-d", type=int, help="Gezinme derinliği.")
    parser.add_argument("--output", "-o", type=str, help="Çıktı dosyasının yolu.")
    parser.add_argument("--max-links", "-m", type=int, help="Her seviyede gezilecek maksimum link sayısı.")
    
    args = parser.parse_args()

    # --- CONFIG ---
    config = load_config()

    # Değerleri belirle (Öncelik sırası: Komut satırı > config dosyası > varsayılan değer)
    base_url = config.get('base_url', 'https://www.jotform.com/help/')
    max_depth = args.depth or config.get('max_depth', 5)
    max_links = args.max_links or config.get('max_links_per_level', 100)
    output_file = args.output or config.get('output_file', 'data/raw/jotform_help_content.txt')
    # --- CONFIG ---

    asyncio.run(main(
        base_url=base_url,
        max_depth=max_depth, 
        output_file=output_file, 
        max_links=max_links
    ))