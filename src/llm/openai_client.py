# src/llm/openai_client.py

import os
import logging
from typing import Dict, Any, Optional
import openai

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class OpenAIClient:
    """
    A simple client to manage communication with the OpenAI API.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("OpenAIClient")

        # A map of model capabilities, primarily for vision support.
        self.model_capabilities = {
            "gpt-4o": {"vision": True},
            "gpt-4o-mini": {"vision": False},  # Small, does not support vision
            "gpt-4.1": {"vision": True},       # Multimodal (text + image)
            "gpt-4.1-mini": {"vision": True},  # Multimodal, fast version
            "gpt-3.5-turbo": {"vision": False}
        }
        
        # Read the API key from environment variables
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            self.logger.error(f"🔴 Environment variable '{api_key_env}' not found. Please set your API key.")
            raise ValueError("OpenAI API key is not set.")
            
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Get model and other settings from the config
        self.model = config.get('model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        
        self.logger.info(f"OpenAI client initialized with model '{self.model}'.")

    def has_vision_capability(self) -> bool:
        """Checks if the currently configured model supports vision."""
        # Look for the model in the capabilities map; return False as a safe default if not found.
        return self.model_capabilities.get(self.model, {}).get("vision", False)

    def get_completion(self, system_prompt: str, user_prompt: str, image_base64: Optional[str] = None) -> str:
        """
        Generates a response from OpenAI using the given prompts.
        """
        self.logger.info("Requesting completion from OpenAI...")
        if image_base64:
            self.logger.info("   - Request includes an image.")

        # The user's content is a list that always includes text.
        user_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]

        # If an image is provided, add it to the content list.
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        elif image_base64 and not self.has_vision_capability():
            self.logger.warning(f"   - Image provided but model '{self.model}' does not support vision. Image will be ignored.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content} # The 'content' is the list we constructed.
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            self.logger.info("Successfully received completion.")
            return content.strip() if content else ""
        except Exception as e:
            self.logger.error(f"An error occurred during the OpenAI API call: {e}")
            return "Sorry, an error occurred and I cannot provide a response at this time."