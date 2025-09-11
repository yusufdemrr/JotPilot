# crawler.py

import asyncio
import os
import argparse
import re
import yaml
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

MIN_SENTENCE_WORDS = 4 # Minimum words in a sentence to be considered meaningful.
SENTENCE_KEY_LENGTH = 50 # Number of characters to consider for duplicate detection.
MIN_CONTENT_LENGTH = 200 # Minimum length of content to be considered valid.
CONTENT_SELECTORS = ['main', 'article', '[role="main"]', '.content', '#content'] # Common selectors for main content.
TAGS_TO_REMOVE = ['nav', 'footer', 'script', 'style', 'aside', 'header'] # Common irrelevant tags.

def clean_text_content(text: str) -> str:
    """
    Cleans text content for RAG purposes.
    - Reduces multiple whitespaces and newlines to a single space.
    - Removes duplicate or short, meaningless sentences.
    """
    # Step 1: Reduce all multiple whitespaces, tabs, and newlines to a single space.
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 2: Split the text into sentences.
    # Splits based on periods, question marks, or exclamation marks followed by a space.
    sentences = re.split(r'(?<=[.?!])\s+', cleaned_text)
    
    unique_sentences = []
    seen_phrases = set() # To store hashes of seen sentences to detect duplicates.
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Skip very short sentences (often menu items or isolated phrases).
        if len(sentence.split()) < MIN_SENTENCE_WORDS:
            continue
            
        # Use the first 50 alphanumeric characters of the sentence as a key
        # to detect and filter out highly similar sentences.
        sentence_key = re.sub(r'[^\w\s]', '', sentence.lower())[:SENTENCE_KEY_LENGTH]
        
        if sentence_key not in seen_phrases:
            seen_phrases.add(sentence_key)
            unique_sentences.append(sentence)
            
    # Rejoin the cleaned and unique sentences.
    return ' '.join(unique_sentences)

def simple_extract_content(html_content: str) -> dict:
    """
    Simply extracts and cleans the main content from an HTML string.
    Returns a dictionary containing the title and content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 1: Find the main page title (h1).
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "Başlık Bulunamadı"
    
    selectors = CONTENT_SELECTORS
    main_content_tag = None
    for selector in selectors:
        main_content_tag = soup.select_one(selector)
        if main_content_tag:
            break
            
    if not main_content_tag:
        main_content_tag = soup.body
        if not main_content_tag:
            return {"title": title, "content": ""}
        
    # Remove the h1 tag from the content to avoid duplication with the title.
    if title_tag and main_content_tag.find('h1'):
        title_tag.decompose()

    # Remove common irrelevant sections like nav, footer, etc.
    for tag in main_content_tag.select(', '.join(TAGS_TO_REMOVE)):
        tag.decompose()
        
    raw_text = main_content_tag.get_text(separator=' ', strip=True)
    
    # Pass the raw text through the cleaning function to remove duplicates and noise.
    cleaned_text = clean_text_content(raw_text)
    
    return {"title": title, "content": cleaned_text}

async def crawl_site(start_url: str, max_depth: int, max_links: int):
    """
    Crawls the specified site, collects content, and returns structured text for each page.
    """
    print(f"Starting crawl. Start URL: {start_url}, Depth: {max_depth}")
    
    all_pages_content = []
    urls_to_visit = {start_url}
    visited_urls = set()
    
    async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
        for depth in range(max_depth + 1):
            if not urls_to_visit:
                break
                
            current_level_urls = list(urls_to_visit)[:max_links]
            urls_to_visit.clear()
            
            print(f"\n--- Level {depth} ---")
            print(f"Found {len(current_level_urls)} URLs to visit.")
            
            for url in current_level_urls:
                if url in visited_urls:
                    continue
                
                print(f"-> Crawling: {url}")
                visited_urls.add(url)
                
                try:
                    result = await crawler.arun(url=url, wait_for="networkidle")
                    
                    if not result.success or not result.html:
                        print(f"  ❌ Failed: {result.error_message}")
                        continue

                    page_data = simple_extract_content(result.html)
                    content = page_data['content']
                    title = page_data['title']
                    
                    if len(content) < MIN_CONTENT_LENGTH:
                        print(f"  ⚠️ Content too short after cleaning ({len(content)} chars), skipping.")
                        continue
                        
                    formatted_content = f"URL: {url}\nTITLE: {title}\n\n{content}"
                    
                    all_pages_content.append(formatted_content)
                    print(f"  ✅ Success: Added page with title '{title}'.")

                    # Find new links to visit for the next level.
                    if depth < max_depth:
                        soup = BeautifulSoup(result.html, 'html.parser')
                        base_domain = urlparse(url).netloc
                        for a_tag in soup.find_all('a', href=True):
                            href = a_tag['href']
                            full_url = urljoin(url, href)
                            if urlparse(full_url).netloc == base_domain and '/help/' in full_url:
                                urls_to_visit.add(full_url)
                                
                except Exception as e:
                    print(f"  ❌ An error occurred: {e}")

    return all_pages_content

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Loads the YAML config file and returns the crawling settings."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                # Just return the 'crawling' section. If not present, return an empty dict.
                return config.get('crawling', {})
            except yaml.YAMLError as e:
                print(f"❌ Error reading the config file: {e}")
    return {}

async def main(base_url: str, max_depth: int, output_file: str, max_links: int):
    """The main function. Initiates the crawling process and writes the result to a file."""
    
    all_content = await crawl_site(base_url, max_depth, max_links)
    
    if not all_content:
        print("\nNo content was found. Terminating process.")
        return

    final_text = "\n\n--- PAGE BREAK ---\n\n".join(all_content)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    print(f"\n🎉 Crawling complete!")
    print(f"📁 Results saved to '{output_file}'.")
    print(f"📄 Processed a total of {len(all_content)} pages.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawls the Jotform Help site and saves the content to a file.")
    parser.add_argument("--depth", "-d", type=int, help="Crawling depth.")
    parser.add_argument("--output", "-o", type=str, help="Path for the output file.")
    parser.add_argument("--max-links", "-m", type=int, help="Maximum number of links to crawl per level.")
    
    args = parser.parse_args()

    # --- CONFIG ---
    config = load_config()

    # Determine the final values (Priority: Command-line args > config file > default values).
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