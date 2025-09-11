# src/api/server.py

import sys
import os
import uuid
from typing import Dict, Any
from dotenv import load_dotenv

# Add project root to the Python path
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
# Import exception handlers from the middleware file
from src.api.middleware import http_exception_handler, general_exception_handler

# --- APPLICATION STARTUP ---
print("🚀 Initializing server and preparing AI Agent...")
agent_brain = ActionAgent()
app = FastAPI(
    title="Jotform AI Agent API",
    version="1.4.0",
    # We use Pydantic aliases to convert Python's snake_case to JSON's camelCase.
    # This provides automatic case conversion at the model level.
)

# --- ADDING MIDDLEWARE AND EXCEPTION HANDLERS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

print("✅ Middlewares (CORS) and exception handlers added.")

# --- IN-MEMORY CACHE ---
SESSION_CACHE: Dict[str, Any] = {}
print("✅ Server, Agent, and in-memory session cache are ready.")


# --- API ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "healthy", "version": app.version}


@app.post("/agent/init", response_model=InitResponse)
async def init_session(request: InitRequest) -> InitResponse:
    """Initializes a new agent session with a given objective."""
    session_id = f"session-{uuid.uuid4()}"
    SESSION_CACHE[session_id] = {
        "objective": request.objective,
        "previous_actions": [],
        "last_proposed_actions": None,
        "last_analyzed_content": None, # For comparing page views
    }

    response = InitResponse(session_id=session_id)
    print(f"✨ New session created: {session_id}")
    return response


@app.post("/agent/next_action", response_model=AgentTurnResponse)
async def next_action(request: AgentTurnRequest) -> AgentTurnResponse:
    """Processes the next turn for an existing agent session."""
    session_id = request.session_id
    print(f"\n▶️  Processing request for session: {session_id}")

    session_data = SESSION_CACHE.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    print(f"   - Updating history based on frontend's report...")
    last_proposed = session_data.get("last_proposed_actions")
    if last_proposed and request.last_turn_outcome:
        # Update the history of executed actions based on the frontend's success/fail report
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
    
    last_analyzed_content = session_data.get("last_analyzed_content")

    # Invoke the agent's brain to decide the next set of actions
    final_state = agent_brain.invoke(
        objective=session_data["objective"],
        visible_elements_html=request.visible_elements_html,
        previous_actions=session_data["previous_actions"],
        user_response=request.user_response,
        screenshot_base64=request.screenshot_base64,
        last_analyzed_content=last_analyzed_content
    )

    response_dict = final_state.get("final_response", {})
    new_actions = response_dict.get("actions", [])
    
    # Cache the newly proposed actions and the latest page analysis for the next turn
    session_data["last_proposed_actions"] = new_actions
    session_data["last_analyzed_content"] = final_state.get("analyzed_content")

    # Construct the response for the frontend
    final_response = AgentTurnResponse(
        session_id=session_id,
        actions=new_actions,
        overall_explanation_of_bundle=response_dict.get(
            "overall_explanation_of_bundle", ""
        ),
        full_thought_process=response_dict.get("full_thought_process"),
        page_summary=response_dict.get("page_summary")
    )
    
    print(f"◀️  Sending response with {len(new_actions)} action(s) for session: {session_id}")
    return final_response