from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Any

class PageAnalyzer:
    """
    Analyzes a list of raw HTML element strings and extracts a structured list.
    Each element is assigned its index from the original list as its primary identifier.
    """

    def __init__(self):
        # A list of selectors to identify any element that a user can interact with.
        self.interactive_selectors = [
            'a[href]', 'button', 'input:not([type=hidden])',
            'textarea', 'select', '[role=button]', '[role=link]'
        ]

    def analyze(self, html_elements: List[str]) -> List[Dict[str, Any]]:
        """
        Parses a list of raw HTML element strings and extracts a simplified list.

        Args:
            html_elements (List[str]): A list of outerHTML strings for interactive elements.

        Returns:
            A list of dictionaries representing these elements with an added index.
        """
        print(f"📊 Analyzing {len(html_elements)} HTML elements...")
        
        interactive_elements = []
        # Use enumerate to get the index, which will serve as our unique ID for this turn.
        for index, html_string in enumerate(html_elements):
            # Each string is a mini-HTML document, so we parse it individually.
            soup = BeautifulSoup(html_string, 'html.parser')
            # The .find() method will get the top-level element from the snippet.
            element = soup.find()
            
            if not element:
                continue

            element_info = {
                "index": index, # The element's position in the list is its ID.
                "text": self._get_element_text(element),
                "selector": self._create_selector(element),
                "tag": element.name,
            }
            interactive_elements.append(element_info)
        
        print(f"👍 Successfully analyzed {len(interactive_elements)} elements.")
        return interactive_elements

    def _get_element_text(self, element: Tag) -> str:
        """Gets the most relevant text from a BeautifulSoup element."""
        aria_label = element.get('aria-label')
        if aria_label:
            return aria_label.strip()
        return element.get_text(strip=True)

    def _create_selector(self, element: Tag) -> str:
        """Creates a robust CSS selector using a priority hierarchy."""
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
            # FIX: Escape quotes *before* creating the f-string to avoid SyntaxError.
            escaped_text = text.replace("'", "\\'").replace('"', '\\"')
            return f"{element.name}:has-text('{escaped_text}')"
            
        return element.name