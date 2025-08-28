# src/web_interaction/page_analyzer.py (DOKÜMAN SIRASI VERSİYONU)

from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Any
import uuid

class PageAnalyzer:
    """
    Analyzes raw HTML content to extract interactive elements.
    It now adds a 'document_order_index' to each element, using the order
    they appear in the HTML source as a proxy for their vertical position.
    """

    def __init__(self):
        self.interactive_selectors = [
            'a[href]', 'button', 'input:not([type=hidden])',
            'textarea', 'select', '[role=button]', '[role=link]'
        ]

    def analyze(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Parses the raw HTML and extracts a simplified list of interactive elements.

        Args:
            html_content (str): The raw HTML of the webpage.

        Returns:
            A list of dictionaries representing interactive elements.
        """
        print("📊 Analyzing HTML to find interactive elements and their document order...")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all candidate elements. BeautifulSoup returns them in document order.
        elements = soup.select(','.join(self.interactive_selectors))
        
        interactive_elements = []
        # Use enumerate to get the index of each element in the list.
        for index, element in enumerate(elements):
            text = self._get_element_text(element)
            
            if text:
                element_info = {
                    "agent_id": str(uuid.uuid4())[:8],
                    "text": text,
                    "selector": self._create_selector(element),
                    "tag": element.name,
                    # --- YENİ: Dokümandaki sırasını (indeksini) ekle ---
                    "document_order_index": index
                }
                interactive_elements.append(element_info)
        
        print(f"👍 Found {len(interactive_elements)} interactive elements.")
        return interactive_elements

    def _get_element_text(self, element: Tag) -> str:
        aria_label = element.get('aria-label')
        if aria_label:
            return aria_label.strip()
        return element.get_text(strip=True)

    def _create_selector(self, element: Tag) -> str:
        if element.get('id'):
            return f"#{element.get('id')}"
        if element.get('data-testid'):
            return f"{element.name}[data-testid='{element.get('data-testid')}']"
        aria_label = element.get('aria-label')
        if aria_label:
             return f"{element.name}[aria-label='{aria_label}']"
        name = element.get('name')
        if name:
            return f"{element.name}[name='{name}']"
        text = self._get_element_text(element)
        if text:
            # Using a simplified escape for quotes.
            # In a production scenario, a more robust CSS escaping library might be needed.
            escaped_text = text.replace("'", "\\'").replace('"', '\\"')
            return f"{element.name}:has-text('{escaped_text}')"
        return element.name