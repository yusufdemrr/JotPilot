# src/agents/action_agent.py

import yaml
import json
from typing import List, Dict, TypedDict, Optional
import re

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

# Import our existing tools and clients
from src.llm.openai_client import OpenAIClient
from src.tools.rag_tool import rag_tool
from src.web_interaction.page_analyzer import PageAnalyzer

class AgentState(TypedDict):
    """
    Represents the state of our agent's thought process in the LangGraph.
    This dictionary is passed between nodes, each node updating parts of it.
    """
    objective: str                  # The main goal from the user.
    visible_elements_html: List[str]      # Raw HTML strings of visible elements on the page.
    analyzed_content: List[Dict]    # The structured analysis of the page content.
    previous_actions: List[Dict]    # A history of actions taken so far.
    rag_context: str                # Relevant info from our knowledge base (fetched by rag_tool).
    final_response: Optional[Dict]  # The final JSON response to be sent to the frontend.
    chat_history: List[BaseMessage] # Not used in this version, but good for future memory.
    user_response: Optional[str]    # Kullanıcıdan gelen cevabı tutar.
    error_feedback: Optional[str] # Yeni state: LLM'e geri bildirim için
    screenshot_base64: Optional[str] # Yeni state: Ekran görüntüsü (base64 formatında), opsiyonel

    retry_count: int          # How many times we've retried this state (for failure recovery)
    last_analyzed_content: Optional[List[Dict]] # Son analiz edilen içerik, page_summary için

class ActionAgent:
    """
    The main brain of our application.
    It uses LangGraph to orchestrate a reasoning loop that:
    1. Retrieves relevant knowledge using the RAG tool.
    2. Thinks and plans the next action(s) based on all available information.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initializes the agent, its tools, and compiles the LangGraph."""
        print("🤖 Initializing ActionAgent (The Brain)...")
        
        # --- 1. Load Configuration and Components ---
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Check if RAG is enabled in features
        self.rag_enabled = self.config.get('features', {}).get('rag_enabled', True)
        print(f"🧠 RAG (Knowledge Base) Enabled: {self.rag_enabled}")

        self.page_analyzer = PageAnalyzer()
        self.openai_client = OpenAIClient(self.config['llm'])
        
        # Load the powerful system prompt for the action agent
        action_prompt_path = self.config['llm']['prompts']['action_system_prompt_path']
        with open(action_prompt_path, 'r', encoding='utf-8') as f:
            self.action_system_prompt = f.read()
            
        # --- 2. Build the LangGraph Workflow ---
        workflow = StateGraph(AgentState)

        # Add the nodes to the graph
        workflow.add_node("analyze_page", self.analyze_page)
        workflow.add_node("retrieve_rag_context", self.retrieve_rag_context)
        workflow.add_node("plan_and_think", self.plan_and_think)
        workflow.add_node("validate_decision", self.validate_decision)

        # Define the entry point of the graph
        workflow.set_entry_point("analyze_page")

        workflow.add_conditional_edges(
            "analyze_page",
            self.should_retrieve_rag,
            {
                "continue_to_rag": "retrieve_rag_context", # RAG aktifse bu yola git
                "skip_rag": "plan_and_think"           # RAG pasifse bu adımı atla, doğrudan düşünmeye geç
            }
        )

        # Define the connections (edges) between the nodes
        workflow.add_edge("retrieve_rag_context", "plan_and_think")
        workflow.add_edge("plan_and_think", "validate_decision")

        # Koşullu kenarı tanımla: Karar geçerli mi, değil mi?
        workflow.add_conditional_edges(
            "validate_decision",
            self.check_decision_validity,
            {
                "valid": END, # Karar geçerliyse bitir
                "invalid": "plan_and_think" # Geçersizse, hata ile tekrar düşün
            }
        )

        # Compile the graph into a runnable object
        self.graph = workflow.compile()
        print("✅ ActionAgent initialized successfully with a compiled LangGraph.")

    # should_retrieve_rag fonksiyonu, config ayarına göre RAG aracını çağırıp çağırmayacağına karar verir
    def should_retrieve_rag(self, state: AgentState) -> str:
        """
        Determines whether to call the RAG tool based on the config setting.
        """
        if self.rag_enabled:
            return "continue_to_rag"
        else:
            return "skip_rag"

    # --- Node 1: Analyze the Current Page Content ---
    def analyze_page(self, state: AgentState) -> Dict:
        """Node 1: Receives raw HTML list and analyzes it."""
        print("--- Node: analyze_page ---")
        analyzed_content = self.page_analyzer.analyze(state["visible_elements_html"])
        return {"analyzed_content": analyzed_content}

    # --- Node 2: Retrieve Context from the Knowledge Base ---
    def retrieve_rag_context(self, state: AgentState) -> Dict:
        """
        This node uses the RAG tool to fetch theoretical knowledge based on the user's objective.
        """
        print("--- Node: retrieve_rag_context ---")
        objective = state["objective"]
        
        # Invoke the RAG tool with the main objective
        rag_response = rag_tool.invoke(objective)
        
        # Return a dictionary to update the state
        return {"rag_context": rag_response}

    # --- Node 3: The Main Brain that Plans the Next Action ---
    def plan_and_think(self, state: AgentState) -> Dict:
        """
        This is the core reasoning node. It gathers all information, constructs a detailed prompt,
        and asks the LLM to decide on the next action(s).
        """
        print("--- Node: plan_and_think ---")

        # Step 1: Prepare the webpage view for the prompt
        analyzed_elements = state['analyzed_content']
        webpage_view_for_prompt = "\n".join([
            f"[index: {el.get('index')}, tag: '{el.get('tag')}', text: '{el.get('text', '')[:100]}...']"
            for el in analyzed_elements
        ])
        
        # Step 2: Prepare the full prompt with ALL context, including any error feedback
        # We use state.get() to safely access 'rag_context'.
        # If the key doesn't exist, it will use the default value instead of crashing.
        # `user_response` hem agent'ın sorusuna cevap hem de kullanıcının ani müdahalesi olabilir.
        user_feedback = state.get('user_response') or 'N/A'
        
        # Hata geri bildirimi varsa, bunu da prompt'a ekliyoruz.
        error_feedback = state.get("error_feedback") or 'N/A. This is the first attempt for this state.'

        # Instructions for page_summary based on whether the page has changed
        current_analyzed_content = state['analyzed_content']
        last_analyzed_content = state.get('last_analyzed_content')
        
        page_has_changed = self._calculate_view_similarity(
            current_view=current_analyzed_content,
            previous_view=last_analyzed_content,
            threshold=0.8 # Eşik değerini doğrudan burada belirleyebiliriz
        )

        if page_has_changed:
            summary_instruction = "The page view has changed. You MUST provide a new, detailed 'page_summary' for the current view."
        else:
            summary_instruction = "The page view has NOT changed significantly. You MUST set 'page_summary' to null or provide a very brief, one-sentence follow-up comment. DO NOT repeat your previous summary."

        prompt_content = f"""
        **High-Level Objective:**
        {state['objective']}

        **Relevant Knowledge from Help Documents (RAG Context):**
        {state.get('rag_context', 'Not used in this turn.')}

        **Current Webpage View (Interactive Elements):**
        {webpage_view_for_prompt}

        **History of Previous Actions:**
        {state['previous_actions']}
        
        **User's Answer to a Previous Question:**
        {user_feedback}
        
        **Feedback on Your Last Attempt (if any):**
        {error_feedback or 'N/A. This is your first attempt.'}

        **Summary Instruction:** 
        {summary_instruction}
        """

        
        # Step 3: Get the decision from the LLM.
        llm_response_str = self.openai_client.get_completion(
            system_prompt=self.action_system_prompt,
            user_prompt=prompt_content,
            image_base64=state.get("screenshot_base64") # Ekran görüntüsünü isteğe bağlı olarak ekle
        )

        print("--- Parsing and Enriching LLM Response ---")
        
        # Step D: Parse the LLM's response and enrich it with the real selector.
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", llm_response_str, re.DOTALL)
            thought_process = thinking_match.group(1).strip() if thinking_match else ""
            
            json_match = re.search(r"<json_response>(.*?)</json_response>", llm_response_str, re.DOTALL)
            if not json_match: raise ValueError("Response missing <json_response> block.")
            
            json_str = json_match.group(1).strip().replace("```json", "").replace("```", "")
            parsed_json = json.loads(json_str)
            
            # Add the thought process to the raw decision payload
            parsed_json["full_thought_process"] = thought_process
            
            print(f"✅ LLM produced a decision draft with {len(parsed_json.get('actions', []))} action(s).")
            
            # The return value is the raw, unvalidated decision from the LLM.
            return {"final_response": parsed_json}

        except Exception as e:
            print(f"❌ ERROR: Failed to parse LLM response: {e}")
            # Create a response that will fail validation
            error_response = {
                "actions": [{ "type": "FAIL", "message": str(e), "explanation": "An internal parsing error occurred." }],
                "overall_explanation_of_bundle": "Failed to process the decision.",
                "full_thought_process": thought_process if 'thought_process' in locals() else "Parsing failed before thoughts could be fully extracted."
            }
            return {"final_response": error_response}
    
    def validate_decision(self, state: AgentState) -> Dict:
        """Node 4: Checks if the index chosen by the LLM is valid."""
        print("--- Node: validate_decision ---")
        actions = state["final_response"].get("actions", [])
        analyzed_content = state["analyzed_content"]

        retry_count = state.get("retry_count", 0)
        MAX_RETRIES = 2 # Maksimum deneme sayısı

        if retry_count >= MAX_RETRIES:
            print(f"🚨 AGENT STUCK: Reached maximum retry limit of {MAX_RETRIES}. Forcing ASK_USER.")
            # Agent'ı döngüden çıkarmak için eylem planını değiştiriyoruz.
            state["final_response"] = {
                "actions": [{
                    "type": "ASK_USER",
                    "user_question": "I seem to be stuck in a loop and can't make a valid decision. Could you please guide me on the next step?",
                    "explanation": "I have failed to make a valid decision multiple times and need user assistance."
                }],
                "overall_explanation_of_bundle": "Requesting user help to resolve a repeating error.",
                "full_thought_process": state["final_response"].get("full_thought_process", "")
            }
            # Döngüyü kırmak için kararın GEÇERLİ olduğunu söylüyoruz.
            return {"error_feedback": None}
        
        if not actions:
            # If no actions, it's a valid (but empty) decision
            return {"error_feedback": None}

        # Check all actions in the bundle for a valid index.
        for action in actions:
            target_index = action.get("target_element_index")
            
            # If an action has a target_element_index, it MUST be valid.
            # Actions like ASK_USER or FINISH might not have an index, which is fine.
            if target_index is not None:
                if not (0 <= target_index < len(analyzed_content)):
                    # If ANY index is invalid, fail the whole bundle and return feedback.
                    print(f"❌ Decision is INVALID. Index {target_index} is out of bounds (0-{len(analyzed_content)-1}).")
                    error = f"Your last decision to use index {target_index} was invalid. The available indices are from 0 to {len(analyzed_content)-1}. Please look at the VIEW again and choose an index that exists in the list."
                    return {"error_feedback": error}
        
        # If the loop completes without finding any invalid indices, the entire bundle is valid.
        print(f"✅ Decision is VALID. All {len(actions)} action(s) have valid indices.")
        return {"error_feedback": None}
        
    def check_decision_validity(self, state: AgentState) -> str:
        """Conditional Edge: Routes to END if valid, or back to plan_and_think if invalid."""
        if state.get("error_feedback"):
            return "invalid"
        else:
            return "valid"
    
    def invoke(self, objective: str, 
               visible_elements_html: List[str], 
               previous_actions: List[Dict], user_response: Optional[str], 
               screenshot_base64: Optional[str], 
               last_analyzed_content: Optional[List[Dict]]) -> Dict:
        """
        The public method to run a single turn of the agent's reasoning loop.
        """
        # This is the initial input for the graph
        inputs = {
            "objective": objective,
            "visible_elements_html": visible_elements_html,
            "previous_actions": previous_actions,
            "chat_history": [], # chat_history is not used yet, but the state requires it
            "user_response": user_response,
            "screenshot_base64": screenshot_base64, # Girdiye ekran görüntüsünü ekle
            "retry_count": 0, # Her yeni 'invoke' çağrısında deneme sayacını sıfırla
            "last_analyzed_content": last_analyzed_content
        }
        
        # Run the graph from start to finish with the given inputs
        final_state = self.graph.invoke(inputs)
        
        # Return the final response calculated by the last node
        return final_state
    
    # Helper function to calculate page similarity
    def _calculate_view_similarity(self, current_view: List[Dict], previous_view: List[Dict], threshold: float = 0.8) -> bool:
        """
        Calculates the Jaccard similarity between the current and previous page views.
        Returns True if the page is considered "changed", False otherwise.
        """
        if not previous_view:
            # If there's no previous view, the page is considered new/changed.
            return True
        
        try:
            # Create a "fingerprint" for each element (e.g., "button:Create Form")
            previous_fingerprints = {f"{el.get('tag')}:{el.get('text', '').strip()}" for el in previous_view}
            current_fingerprints = {f"{el.get('tag')}:{el.get('text', '').strip()}" for el in current_view}
            
            # Calculate Jaccard similarity
            intersection = len(previous_fingerprints.intersection(current_fingerprints))
            union = len(previous_fingerprints.union(current_fingerprints))
            
            similarity = intersection / union if union > 0 else 1.0
            
            print(f"   - Page similarity score: {similarity:.2f} (Threshold: {threshold})")
            
            # If similarity is below the threshold, the page has changed.
            return similarity < threshold

        except Exception as e:
            print(f"⚠️ Could not calculate page similarity: {e}")
            # In case of error, assume the page has changed to be safe.
            return True
