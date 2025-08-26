# src/llm/openai_client.py

import os
import logging
from typing import Dict, Any
import openai

# Temel loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class OpenAIClient:
    """
    OpenAI API'si ile iletişimi yöneten basit bir istemci.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("OpenAIClient")
        
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

    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Verilen prompt'lar ile OpenAI'den bir cevap üretir.
        """
        self.logger.info("OpenAI'den cevap isteniyor...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            self.logger.info("Cevap başarıyla alındı.")
            return content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API çağrısı sırasında bir hata oluştu: {e}")
            return "Üzgünüm, bir hata oluştu ve şu anda cevap veremiyorum."