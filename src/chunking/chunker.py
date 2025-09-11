# chunker.py

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
    A simple system that splits text from the crawler into chunks for RAG.
    This version is simplified and adapted for the new crawler output format
    which uses '--- PAGE BREAK ---' as a delimiter.
    """
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the chunker with basic parameters.
        
        Args:
            chunk_size (int): The target token size for each chunk.
            chunk_overlap (int): The token overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
            print(f"✅ Tokenizer loaded for model: {model_name}")
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            print(f"⚠️ Warning: Model '{model_name}' not found. Using default encoder 'cl100k_base'.")
            
        print(f"✅ SimpleChunker initialized. Target size: {chunk_size} tokens, Overlap: {chunk_overlap} tokens.")

    def count_tokens(self, text: str) -> int:
        """Returns the token count for a given text."""
        return len(self.encoder.encode(text))

    def parse_pages_from_txt(self, txt_file_path: str) -> List[Dict[str, str]]:
        """
        Parses the crawler's output file, which contains URL, TITLE, and content for each page.
        """
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Input file not found -> {txt_file_path}")
            return []

        pages = []
        raw_pages = content.split('--- PAGE BREAK ---')
        
        for raw_page in raw_pages:
            raw_page = raw_page.strip()
            if not raw_page:
                continue
            
            lines = raw_page.split('\n')
            if len(lines) < 3:  # Expect at least URL, TITLE, and some content.
                continue
            
            url_line = lines[0]
            title_line = lines[1]
            
            url_match = re.match(r"URL: (https?://[^\s]+)", url_line)
            title_match = re.match(r"TITLE: (.+)", title_line)

            if url_match and title_match:
                url = url_match.group(1).strip()
                title = title_match.group(1).strip()
                page_content = '\n'.join(lines[2:]).strip()
                pages.append({"url": url, "title": title, "content": page_content})
        
        print(f"📄 Successfully parsed {len(pages)} pages.")
        return pages

    def create_chunks(self, pages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Creates chunks from the parsed pages.
        """
        all_chunks = []
        
        # Using LangChain's effective text splitter is simpler and more standard
        # than writing a custom splitting function.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", " ", ""] # Priority order for splitting
        )
        
        for page in pages:
            chunks_content = text_splitter.split_text(page['content'])
            
            for i, chunk_text in enumerate(chunks_content):
                chunk = {
                    "id": str(uuid.uuid4()), # Unique identifier for each chunk
                    "content": chunk_text,
                    "metadata": {
                        "source_url": page['url'],
                        "title": page['title'],
                        "chunk_index": i + 1,
                        "token_count": self.count_tokens(chunk_text)
                    }
                }
                all_chunks.append(chunk)
        
        print(f"🎉 Created a total of {len(all_chunks)} chunks.")
        return all_chunks

    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_file: str):
        """Saves the generated chunks to a JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Chunks successfully saved to '{output_file}'.")

def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """Loads the YAML config file."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                print(f"📋 Loading settings from: {config_path}")
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"❌ Error reading the config file: {e}")
    return {}

def main():
    parser = argparse.ArgumentParser(description="Chunks a text file for RAG.")
    parser.add_argument("input_file", help="The input .txt file generated by the crawler.")
    parser.add_argument("--output", "-o", help="Path for the output JSON file.")
    parser.add_argument("--chunk-size", "-s", type=int, help="Target chunk size in tokens.")
    parser.add_argument("--chunk-overlap", "-v", type=int, help="Chunk overlap in tokens.")

    args = parser.parse_args()

    # --- CONFIGURATION ---
    config = load_config()
    chunking_config = config.get('chunking', {})

    # Determine values (Priority: Command-line > config file > hardcoded default)
    input_file = args.input_file # This is mandatory from argparse
    output_file = args.output or chunking_config.get('output_file', 'data/chunks/chunks.json')
    chunk_size = args.chunk_size or chunking_config.get('target_chunk_size', 400)
    chunk_overlap = args.chunk_overlap or chunking_config.get('overlap_size', 50)
    model_name = chunking_config.get('tokenizer_model', 'gpt-3.5-turbo')
    # --- CONFIGURATION ---

    chunker = SimpleChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name
    )

    pages = chunker.parse_pages_from_txt(input_file)

    if pages:
        chunks = chunker.create_chunks(pages)
        if chunks:
            chunker.save_chunks_to_json(chunks, output_file)

if __name__ == "__main__":
    main()