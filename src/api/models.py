# src/api/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- /agent/init Endpoint Modelleri ---


class InitRequest(BaseModel):
    objective: str = Field(..., description="Kullanıcının bu oturumdaki ana hedefi.")


class InitResponse(BaseModel):
    session_id: str = Field(..., description="Oturum için oluşturulan benzersiz ID.")


# --- /agent/next_action Endpoint Modelleri ---

# --------------------------------------------------------------------------
# Frontend'den Backend'e Gönderilecek Veri Modelleri (API Request)
# --------------------------------------------------------------------------


class ExecutedAction(BaseModel):
    """Frontend, bir önceki turun sonucunu bu basit modelle raporlar."""

    status: str = Field(..., description="'SUCCESS' veya 'FAIL'")
    error_message: Optional[str] = None  # Eğer FAIL ise


class AgentTurnRequest(BaseModel):
    session_id: str
    visible_elements_html: List[str]
    user_response: Optional[str] = None

    # Frontend artık sadece bir önceki turun sonucunu gönderir, detayları değil.
    last_turn_outcome: List[ExecutedAction] = Field(
        ...,
        description="Bir önceki turda gerçekleştirilen eylemlerin başarı/hata durumu.",
    )

    # Optional: Eğer vision_enabled ise, frontend ekran görüntüsünü de gönderebilir 
    screenshot_base64: Optional[str] = Field(None, description="Sayfanın o anki ekran görüntüsü (base64 formatında). Opsiyonel.")

# ActionHistory artık sadece backend'in dahili olarak kullandığı bir yapı
class ActionHistory(BaseModel):
    action_type: str
    description: str


# --------------------------------------------------------------------------
# Backend'den Frontend'e Gönderilecek Veri Modelleri (API Response)
# --------------------------------------------------------------------------


class Action(BaseModel):
    """
    Represents a single, concrete command for the frontend to execute.
    """

    type: str = Field(
        ...,
        description="The type of action: 'CLICK', 'TYPE', 'ASK_USER', 'FINISH', 'FAIL'.",
    )

    # The key field: the index of the target element in the list the frontend sent.
    target_element_index: Optional[int] = Field(
        None,
        description="The index of the target element within the 'visible_elements_html' list.",
    )

    type_value: Optional[str] = Field(None, description="The text to be typed for TYPE actions.")
    user_question: Optional[str] = Field(None, description="The question to ask the user for ASK_USER actions.")
    status_message: Optional[str] = Field(None, description="The final status message for FINISH or FAIL actions.")
    explanation: str


class AgentTurnResponse(BaseModel):
    """
    The complete response package the backend sends to the frontend.
    It contains a list of actions to perform.
    """

    session_id: str
    actions: List[Action]
    overall_explanation_of_bundle: str
    full_thought_process: Optional[str] = None
