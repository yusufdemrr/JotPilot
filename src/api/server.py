# src/api/server.py (NİHAİ VE EN SAĞLAM VERSİYON)

import sys
import os
import json
import uuid
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv('config/.env')

from fastapi import FastAPI, HTTPException
from src.agents.action_agent import ActionAgent
from src.api.models import InitRequest, InitResponse, AgentTurnRequest, AgentTurnResponse, ExecutedAction
from typing import Dict, Any

# --- UYGULAMA BAŞLANGICI ---
print("🚀 Sunucu başlatılıyor ve AI Agent hazırlanıyor...")
agent_brain = ActionAgent()
app = FastAPI(title="Jotform AI Agent API", version="1.3.0") # Sürümü güncelleyelim

SESSION_CACHE: Dict[str, Any] = {}
print("✅ Sunucu, Agent ve in-memory session cache hazır.")

# --- API ENDPOINTS ---

@app.post("/agent/init", response_model=InitResponse)
async def init_session(request: InitRequest) -> InitResponse:
    session_id = f"session-{uuid.uuid4()}"
    SESSION_CACHE[session_id] = {
        "objective": request.objective,
        "previous_actions": [], # Başarıyla tamamlanmış eylemlerin geçmişi
        "last_proposed_actions": None # Son önerilen ama henüz onaylanmamış eylemler
    }
    print(f"✨ New session created: {session_id}")
    return InitResponse(session_id=session_id)


@app.post("/agent/next_action", response_model=AgentTurnResponse)
async def next_action(request: AgentTurnRequest) -> AgentTurnResponse:
    session_id = request.session_id
    print(f"\n▶️  Received request for session: {session_id}")

    session_data = SESSION_CACHE.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Adım 1: Frontend'in raporuna göre RESMİ GEÇMİŞİ GÜNCELLE
    print(f"   - Updating history based on frontend's report...")
    last_proposed = session_data.get("last_proposed_actions")
    if last_proposed and request.last_turn_outcome:
        for i, outcome in enumerate(request.last_turn_outcome):
            action_to_log = last_proposed[i] # Önerdiğimiz eylemi al
            if outcome.status.upper() == 'SUCCESS':
                session_data["previous_actions"].append({
                    "action_type": action_to_log.get("type"),
                    "description": action_to_log.get("explanation")
                })
            else:
                session_data["previous_actions"].append({
                    "action_type": "FAIL",
                    "description": f"Action '{action_to_log.get('type')}' failed with error: {outcome.error_message}"
                })
    
    # Adım 2: Agent'ın beynini, güncel ve doğru geçmişle çağır
    final_state = agent_brain.invoke(
        objective=session_data["objective"],
        visible_elements_html=request.visible_elements_html,
        previous_actions=session_data["previous_actions"],
        user_response=request.user_response
    )
    
    response_dict = final_state.get("final_response", {})
    new_actions = response_dict.get("actions", [])

    # Adım 3: Yeni önerilen eylemleri, bir sonraki turda doğrulamak için CACHE'E KAYDET
    session_data["last_proposed_actions"] = new_actions

    # Adım 4: Frontend'e cevabı formatla ve gönder
    final_response = AgentTurnResponse(
        session_id=session_id,
        actions=new_actions,
        overall_explanation_of_bundle=response_dict.get("overall_explanation_of_bundle", ""),
        full_thought_process=response_dict.get("full_thought_process")
    )

    print(f"◀️  Sending response for session: {session_id}")
    return final_response