# src/llm/chatbot_langgraph.py

import yaml
import os
from typing import List, TypedDict
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Diğer modüllerimizi import edelim
from src.llm.openai_client import OpenAIClient
from src.embedding.qdrant_manager import QdrantManager
from dotenv import load_dotenv

dotenv_path = 'config/.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Adım 1: Grafiğin "Hafızasını" (State) Tanımla ---
# State, grafikteki adımlar (node'lar) arasında taşınan veri yapısıdır.
class GraphState(TypedDict):
    query: str                      # Kullanıcının son sorusu
    context: str                    # Qdrant'tan çekilen bilgi
    response: str                   # LLM tarafından üretilen son cevap
    chat_history: List[BaseMessage] # Tüm konuşma geçmişi (LangChain formatında)


class LangGraphChatbot:
    """
    Tüm RAG sürecini LangGraph ile yöneten, ölçeklenebilir ve modern chatbot.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        print("🤖 LangGraph Chatbot başlatılıyor...")
        
        # --- Bileşenleri Başlat ---
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.openai_client = OpenAIClient(self.config['llm'])
        self.qdrant_manager = QdrantManager(self.config['qdrant'])
        
        model_name = self.config['models']['primary']['model_name']
        print(f"'{model_name}' embedding modeli yükleniyor...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load prompt templates from the file paths
        try:
            system_prompt_path = self.config['llm']['prompts']['rag_system_prompt_path']
            rag_template_path = self.config['llm']['prompts']['rag_template_path']

            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()

            with open(rag_template_path, 'r', encoding='utf-8') as f:
                self.rag_template = f.read()
        
        except FileNotFoundError as e:
            print(f"❌ ERROR: Prompt file not found: {e}")
            raise
        except KeyError as e:
            print(f"❌ ERROR: Prompt path key not found in config.yaml: {e}")
            raise
        
        # --- Adım 2: LangGraph Akış Şemasını Oluştur ---
        workflow = StateGraph(GraphState)

        # Grafiğe adımları (node'ları) ekle
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)

        # Grafiğin başlangıç noktasını belirle
        workflow.set_entry_point("retrieve_context")

        # Adımlar arasındaki bağlantıları (edge'leri) kur
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END) # 'generate_response' adımı sondur

        # Hazırlanan akış şemasını derle ve çalıştırılabilir hale getir
        self.graph = workflow.compile()
        print("✅ LangGraph Chatbot başarıyla başlatıldı ve çalışmaya hazır.")

    # --- Adım 3: Grafik Düğümlerini (Node) Fonksiyon Olarak Tanımla ---

    def retrieve_context(self, state: GraphState) -> dict:
        """
        Node 1: Kullanıcının sorusunu alır, Qdrant'ta arama yapar ve context'i bulur.
        """
        print("-> Node: retrieve_context çalışıyor...")
        query = state["query"]
        
        query_vector = self.embedding_model.encode(query).tolist()
        
        search_results = self.qdrant_manager.search(
            query_vector=query_vector,
            limit=self.config['llm']['rag']['search_limit']
        )
        
        if not search_results:
            context = "Bu konu hakkında spesifik bir bilgi bulunamadı."
        else:
            context_parts = [result.payload['content'] for result in search_results]
            context = "\n\n---\n\n".join(context_parts)
            
        return {"context": context}

    def generate_response(self, state: GraphState) -> dict:
        """
        Node 2: Geçmişi, context'i ve soruyu alıp LLM'e gönderir, cevabı üretir.
        """
        print("-> Node: generate_response çalışıyor...")
        query = state["query"]
        context = state["context"]
        chat_history = state["chat_history"]

        # Konuşma geçmişini basit bir metne dönüştür
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        
        prompt_with_context = (
            f"Önceki Konuşma Geçmişi:\n{history_str}\n\n"
            f"Şimdiki Soru: {query}"
        )

        final_prompt = self.rag_template.format(
            context=context,
            question=prompt_with_context
        )

        response = self.openai_client.get_completion(
            system_prompt=self.system_prompt,
            user_prompt=final_prompt
        )
        
        return {"response": response}

    def invoke(self, query: str, chat_history: List[BaseMessage]) -> str:
        """
        Grafiği çalıştırır ve nihai cevabı döndürür.
        """
        inputs = {"query": query, "chat_history": chat_history}
        # Grafiği bu başlangıç durumuyla çalıştır
        final_state = self.graph.invoke(inputs)
        return final_state["response"]

def main():
    """Chatbot'u interaktif modda çalıştırır."""

    chatbot = LangGraphChatbot()
    
    # Hafıza, ana döngüde tutulur
    chat_history = []
    max_history = chatbot.config.get('chat', {}).get('max_history', 5)

    print("\n--- Jotform AI Agent ---")
    print("Merhaba! Jotform hakkında ne öğrenmek istersiniz? (Çıkmak için 'exit' yazın)")
    
    while True:
        user_query = input("\n👤 You: ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Bye!")
            break
            
        # Grafiği mevcut geçmişle birlikte çalıştır
        response = chatbot.invoke(user_query, chat_history)
        
        print(f"\n🤖 Agent: {response}")
        
        # Hafızayı güncelle
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=response))
        
        # Hafızanın çok uzamasını engelle
        if len(chat_history) > max_history * 2:
            chat_history = chat_history[-(max_history * 2):]

if __name__ == "__main__":
    main()