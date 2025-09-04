# src/llm/openai_client.py

import os
import logging
from typing import Dict, Any, Optional
import openai

# Temel loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class OpenAIClient:
    """
    OpenAI API'si ile iletişimi yöneten basit bir istemci.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("OpenAIClient")

        self.model_capabilities = {
            "gpt-4o": {"vision": True},
            "gpt-4o-mini": {"vision": False}, # Daha küçük, görsel desteklemeyen model
            "gpt-3.5-turbo": {"vision": False}
        }
        
        # API anahtarını ortam değişkenlerinden (environment variable) oku
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            self.logger.error(f"🔴 '{api_key_env}' ortam değişkeni bulunamadı. Lütfen API anahtarınızı ayarlayın.")
            raise ValueError("OpenAI API anahtarı ayarlanmamış.")
            
        # OpenAI istemcisini başlat
        self.client = openai.OpenAI(api_key=api_key)
        
        # Model ve diğer ayarları config'den al
        self.model = config.get('model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        
        self.logger.info(f"OpenAI istemcisi '{self.model}' modeli ile başlatıldı.")

    def has_vision_capability(self) -> bool:
        """Checks if the currently configured model supports vision."""
        # Modeli yetenek haritasında ara, bulamazsan güvenli olması için False döndür.
        return self.model_capabilities.get(self.model, {}).get("vision", False)

    def get_completion(self, system_prompt: str, user_prompt: str, image_base64: Optional[str] = None) -> str:
        """
        Verilen prompt'lar ile OpenAI'den bir cevap üretir.
        """
        self.logger.info("OpenAI'den cevap isteniyor...")
        if image_base64:
            self.logger.info("   - Request includes an image.")

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
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    # The 'content' is now the list we constructed.
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            self.logger.info("Successfully received completion.")
            return content.strip() if content else ""
        except Exception as e:
            self.logger.error(f"OpenAI API çağrısı sırasında bir hata oluştu: {e}")
            return "Üzgünüm, bir hata oluştu ve şu anda cevap veremiyorum."