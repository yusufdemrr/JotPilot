# src/api/server.py (TEMİZLENMİŞ VE MODÜLER VERSİYON)

import sys
import os
import uuid
import json
from typing import Dict, Any
from dotenv import load_dotenv

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
load_dotenv("config/.env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.agents.action_agent import ActionAgent
from src.api.models import (
    InitRequest,
    InitResponse,
    AgentTurnRequest,
    AgentTurnResponse,
)

# Hata yöneticilerini middleware dosyasından import ediyoruz
from src.api.middleware import http_exception_handler, general_exception_handler

# --- UYGULAMA BAŞLANGICI ---
print("🚀 Sunucu başlatılıyor ve AI Agent hazırlanıyor...")
agent_brain = ActionAgent()
app = FastAPI(
    title="Jotform AI Agent API",
    version="1.4.0",
    # Python'un snake_case'ini JSON'un camelCase'ine çevirmek için Pydantic alias'ları kullanıyoruz
    # Bu, model seviyesinde otomatik case conversion sağlar.
)

# --- MIDDLEWARE ve HATA YÖNETİCİLERİNİ EKLEME ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

print("✅ Middleware'ler (CORS) ve hata yöneticileri eklendi.")

# --- IN-MEMORY CACHE ---
SESSION_CACHE: Dict[str, Any] = {}
print("✅ Sunucu, Agent ve in-memory session cache hazır.")

# --- API ENDPOINTS ---


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": app.version}


@app.post("/agent/init", response_model=InitResponse)
async def init_session(request: InitRequest) -> InitResponse:
    # Print request payload as raw JSON

    session_id = f"session-{uuid.uuid4()}"
    SESSION_CACHE[session_id] = {
        "objective": request.objective,
        "previous_actions": [],
        "last_proposed_actions": None,
    }

    response = InitResponse(session_id=session_id)

    print(f"✨ New session created: {session_id}")

    return response


@app.post("/agent/next_action", response_model=AgentTurnResponse)
async def next_action(request: AgentTurnRequest) -> AgentTurnResponse:
    session_id = request.session_id

    print(f"\n▶️  Processing request for session: {session_id}")

    session_data = SESSION_CACHE.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    print(f"   - Updating history based on frontend's report...")
    last_proposed = session_data.get("last_proposed_actions")
    if last_proposed and request.last_turn_outcome:
        for i, outcome in enumerate(request.last_turn_outcome):
            action_to_log = last_proposed[i]
            if outcome.status.upper() == "SUCCESS":
                session_data["previous_actions"].append(
                    {
                        "action_type": action_to_log.get("type"),
                        "description": action_to_log.get("explanation"),
                    }
                )
            else:
                session_data["previous_actions"].append(
                    {
                        "action_type": "FAIL",
                        "description": f"Action '{action_to_log.get('type')}' failed with error: {outcome.error_message}",
                    }
                )

    final_state = agent_brain.invoke(
        objective=session_data["objective"],
        visible_elements_html=request.visible_elements_html,
        previous_actions=session_data["previous_actions"],
        user_response=request.user_response,
        screenshot_base64=request.screenshot_base64,
    )

    response_dict = final_state.get("final_response", {})
    new_actions = response_dict.get("actions", [])
    session_data["last_proposed_actions"] = new_actions

    final_response = AgentTurnResponse(
        session_id=session_id,
        actions=new_actions,
        overall_explanation_of_bundle=response_dict.get(
            "overall_explanation_of_bundle", ""
        ),
        full_thought_process=response_dict.get("full_thought_process"),
        page_summary=response_dict.get("page_summary")
    )

    print(f"◀️  Sending response for session: {session_id}")
    return final_response
