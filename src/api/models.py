# src/api/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- /agent/init Endpoint Models ---


class InitRequest(BaseModel):
    objective: str = Field(..., description="Kullanıcının bu oturumdaki ana hedefi.")


class InitResponse(BaseModel):
    session_id: str = Field(
        ..., alias="sessionId", description="Oturum için oluşturulan benzersiz ID."
    )

    class Config:
        populate_by_name = True


# --- /agent/next_action Endpoint Models ---

# --------------------------------------------------------------------------
# Data Models Sent from Frontend to Backend (API Request)
# --------------------------------------------------------------------------


class ExecutedAction(BaseModel):
    """Reports the result of the previous turn from the frontend."""

    status: str = Field(..., description="'SUCCESS' veya 'FAIL'")
    error_message: Optional[str] = Field(None, alias="errorMessage")  # if status is FAIL

    class Config:
        populate_by_name = True


class AgentTurnRequest(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    visible_elements_html: List[str] = Field(..., alias="visibleElementsHtml")
    user_response: Optional[str] = Field(None, alias="userResponse")

    # The frontend now only sends the outcome of the previous turn, not all the details.
    last_turn_outcome: List[ExecutedAction] = Field(
        ...,
        alias="lastTurnOutcome",
        description="Bir önceki turda gerçekleştirilen eylemlerin başarı/hata durumu.",
    )

    class Config:
        populate_by_name = True

    # Optional: If vision is enabled, the frontend can also send a screenshot.
    screenshot_base64: Optional[str] = Field(
        None,
        alias="screenshotBase64",
        description="Sayfanın o anki ekran görüntüsü (base64 formatında). Opsiyonel.",
    )


# ActionHistory is now an internal structure used only by the backend.
class ActionHistory(BaseModel):
    action_type: str = Field(..., alias="actionType")
    description: str

    class Config:
        populate_by_name = True


# --------------------------------------------------------------------------
# Data Models Sent from Backend to Frontend (API Response)
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
        alias="targetElementIndex",
        description="The index of the target element within the 'visible_elements_html' list.",
    )

    type_value: Optional[str] = Field(
        None, alias="typeValue", description="The text to be typed for TYPE actions."
    )
    user_question: Optional[str] = Field(
        None,
        alias="userQuestion",
        description="The question to ask the user for ASK_USER actions.",
    )
    status_message: Optional[str] = Field(
        None,
        alias="statusMessage",
        description="The final status message for FINISH or FAIL actions.",
    )
    explanation: str

    class Config:
        populate_by_name = True


class AgentTurnResponse(BaseModel):
    """
    The complete response package the backend sends to the frontend.
    It contains a list of actions to perform.
    """

    session_id: str = Field(..., alias="sessionId")
    actions: List[Action]
    overall_explanation_of_bundle: str = Field(..., alias="overallExplanationOfBundle")
    full_thought_process: Optional[str] = Field(None, alias="fullThoughtProcess")
    # For educational purposes, the backend can optionally provide a general summary of the page.
    page_summary: Optional[str] = Field(None, alias="pageSummary", description="O anki sayfa hakkında genel, eğitici bir özet veya ipucu.")
    
    class Config:
        populate_by_name = True
