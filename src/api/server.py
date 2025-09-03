# src/api/server.py (NİHAİ VE EN SAĞLAM VERSİYON)

import sys
import os
import json
import uuid
from dotenv import load_dotenv

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
load_dotenv("config/.env")

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from src.agents.action_agent import ActionAgent
from src.api.models import (
    InitRequest,
    InitResponse,
    AgentTurnRequest,
    AgentTurnResponse,
    ExecutedAction,
    Action,
)
from typing import Dict, Any, List, Optional
import re

# --- UYGULAMA BAŞLANGICI ---
print("🚀 Sunucu başlatılıyor ve AI Agent hazırlanıyor...")
agent_brain = ActionAgent()
app = FastAPI(title="Jotform AI Agent API", version="1.3.0")  # Sürümü güncelleyelim

SESSION_CACHE: Dict[str, Any] = {}
print("✅ Sunucu, Agent ve in-memory session cache hazır.")


# --- CASE CONVERSION MIDDLEWARE ---
def to_camel_case(snake_str):
    """Convert snake_case to camelCase"""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def to_snake_case(camel_str):
    """Convert camelCase to snake_case"""
    # Use regex to find all capital letters and add underscore before them
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def convert_dict_keys_to_camel_case(d):
    """Recursively convert all dictionary keys from snake_case to camelCase"""
    if not isinstance(d, dict):
        return d

    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_dict_keys_to_camel_case(v)
        elif isinstance(v, list):
            v = [
                (
                    convert_dict_keys_to_camel_case(item)
                    if isinstance(item, (dict, list))
                    else item
                )
                for item in v
            ]

        # Convert key to camelCase
        camel_key = to_camel_case(k)
        result[camel_key] = v

    return result


def convert_dict_keys_to_snake_case(d):
    """Recursively convert all dictionary keys from camelCase to snake_case"""
    if not isinstance(d, dict):
        return d

    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_dict_keys_to_snake_case(v)
        elif isinstance(v, list):
            v = [
                (
                    convert_dict_keys_to_snake_case(item)
                    if isinstance(item, (dict, list))
                    else item
                )
                for item in v
            ]

        # Convert key to snake_case
        snake_key = to_snake_case(k)
        result[snake_key] = v

    return result


class CaseConversionMiddleware(BaseHTTPMiddleware):
    """Middleware to convert between snake_case and camelCase"""

    async def dispatch(self, request: Request, call_next):
        # Convert request body from camelCase to snake_case if it's a JSON request
        if request.headers.get("content-type") == "application/json":
            try:
                # Read the request body
                body = await request.body()
                data = json.loads(body)

                # Convert camelCase to snake_case for request
                data = convert_dict_keys_to_snake_case(data)

                # Create a new request with the converted data
                # We need to modify the request._receive to return the new body
                async def receive():
                    return {"type": "http.request", "body": json.dumps(data).encode()}

                request._receive = receive
            except Exception as e:
                print(f"Error converting request to snake_case: {e}")
                # Continue with the original request if there's an error
                pass

        # Process the request
        response = await call_next(request)

        # Convert response body from snake_case to camelCase if it's a JSON response
        if response.headers.get("content-type") == "application/json":
            # Get the response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Parse the JSON body
            try:
                data = json.loads(body.decode())
                # Convert snake_case to camelCase
                data = convert_dict_keys_to_camel_case(data)
                # Create a new response with the converted data
                # Remove Content-Length header to let FastAPI calculate it correctly
                headers = dict(response.headers)
                if "content-length" in headers:
                    del headers["content-length"]

                return JSONResponse(
                    content=data,
                    status_code=response.status_code,
                    headers=headers,
                )
            except Exception as e:
                # If there's an error, return the original response
                print(f"Error converting response to camelCase: {e}")
                # Remove Content-Length header to let FastAPI calculate it correctly
                headers = dict(response.headers)
                if "content-length" in headers:
                    del headers["content-length"]

                return Response(
                    body,
                    status_code=response.status_code,
                    headers=headers,
                )

        return response


# Add the middleware to the application
app.add_middleware(CaseConversionMiddleware)
print("✅ Added case conversion middleware for API compatibility")
print("   - Requests: camelCase → snake_case (for backend compatibility)")
print("   - Responses: snake_case → camelCase (for frontend compatibility)")
print("   - No model changes required, conversion happens at HTTP layer")


# --- ERROR HANDLING ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "code": f"HTTP_{exc.status_code}_ERROR",
            "details": {},
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": str(exc), "code": "INTERNAL_SERVER_ERROR", "details": {}},
    )


# --- API ENDPOINTS ---


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.3.0"}


@app.post("/agent/init", response_model=InitResponse)
async def init_session(request: InitRequest) -> InitResponse:
    session_id = f"session-{uuid.uuid4()}"
    SESSION_CACHE[session_id] = {
        "objective": request.objective,
        "previous_actions": [],  # Başarıyla tamamlanmış eylemlerin geçmişi
        "last_proposed_actions": None,  # Son önerilen ama henüz onaylanmamış eylemler
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
            action_to_log = last_proposed[i]  # Önerdiğimiz eylemi al
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

    # Adım 2: Agent'ın beynini, güncel ve doğru geçmişle çağır
    final_state = agent_brain.invoke(
        objective=session_data["objective"],
        visible_elements_html=request.visible_elements_html,
        previous_actions=session_data["previous_actions"],
        user_response=request.user_response,
    )

    response_dict = final_state.get("final_response", {})
    new_actions = response_dict.get("actions", [])

    # Adım 3: Yeni önerilen eylemleri, bir sonraki turda doğrulamak için CACHE'E KAYDET
    session_data["last_proposed_actions"] = new_actions

    # Adım 4: Frontend'e cevabı formatla ve gönder
    final_response = AgentTurnResponse(
        session_id=session_id,
        actions=new_actions,
        overall_explanation_of_bundle=response_dict.get(
            "overall_explanation_of_bundle", ""
        ),
        full_thought_process=response_dict.get("full_thought_process"),
    )

    print(f"◀️  Sending response for session: {session_id}")
    return final_response
