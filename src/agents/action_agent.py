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

class AgentState(TypedDict):
    """
    Represents the state of our agent's thought process in the LangGraph.
    This dictionary is passed between nodes, each node updating parts of it.
    """
    objective: str                  # The main goal from the user.
    current_page_content: str       # The simplified HTML from PageAnalyzer.
    previous_actions: List[Dict]    # A history of actions taken so far.
    rag_context: str                # Relevant info from our knowledge base (fetched by rag_tool).
    final_response: Optional[Dict]  # The final JSON response to be sent to the frontend.
    chat_history: List[BaseMessage] # Not used in this version, but good for future memory.
    user_response: Optional[str]    # Kullanıcıdan gelen cevabı tutar.

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

        self.openai_client = OpenAIClient(self.config['llm'])
        self.tools = [rag_tool] # A list of tools the agent can use. For now, just one.
        
        # Load the powerful system prompt for the action agent
        action_prompt_path = self.config['llm']['prompts']['action_system_prompt_path']
        with open(action_prompt_path, 'r', encoding='utf-8') as f:
            self.action_system_prompt = f.read()
            
        # --- 2. Build the LangGraph Workflow ---
        workflow = StateGraph(AgentState)

        # Add the nodes to the graph
        workflow.add_node("retrieve_rag_context", self.retrieve_rag_context)
        workflow.add_node("plan_and_think", self.plan_and_think)

        # Define the entry point of the graph
        workflow.set_entry_point("retrieve_rag_context")

        # Define the connections (edges) between the nodes
        workflow.add_edge("retrieve_rag_context", "plan_and_think")
        workflow.add_edge("plan_and_think", END) # The 'plan_and_think' node is the last step in a single turn.

        # Compile the graph into a runnable object
        self.graph = workflow.compile()
        print("✅ ActionAgent initialized successfully with a compiled LangGraph.")

    # --- Node 1: Retrieve Context from the Knowledge Base ---
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

    # --- Node 2: The Main Brain that Plans the Next Action ---
    def plan_and_think(self, state: AgentState) -> Dict:
        """
        This is the core reasoning node. It gathers all information, constructs a detailed prompt,
        and asks the LLM to decide on the next action(s).
        """
        print("--- Node: plan_and_think ---")
        
        # Prepare the prompt content by formatting all available information
        prompt_content = f"""
        **High-Level Objective:**
        {state['objective']}

        **Relevant Knowledge from Help Documents (RAG Context):**
        {state['rag_context']}

        **Current Webpage View (Interactive Elements):**
        {state['current_page_content']}

        **History of Previous Actions:**
        {state['previous_actions']}
        
        **User's Answer to a Previous Question:**
        {state.get('user_response') or 'N/A'} 
        """

        
        # Get the final answer from the LLM
        llm_response_str = self.openai_client.get_completion(
            system_prompt=self.action_system_prompt,
            user_prompt=prompt_content
        )

        print("--- Parsing LLM Response ---")
        
        # --- YENİ DÜŞÜNCE AYRIŞTIRMA MANTIĞI ---
        
        # 1. Use regex to extract the thinking process
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", llm_response_str, re.DOTALL)
        thought_process = thinking_match.group(1).strip() if thinking_match else ""
        
        # 2. Use regex to extract the JSON response
        json_match = re.search(r"<json_response>(.*?)</json_response>", llm_response_str, re.DOTALL)
        
        final_response_payload = {}

        if json_match:
            json_str = json_match.group(1).strip()
            # Clean up potential markdown code blocks around the JSON
            if json_str.startswith("```json"):
                json_str = json_str[7:-4].strip()
            
            try:
                parsed_json = json.loads(json_str)
                
                # --- YENİ: agent_id'yi gerçek seçiciye dönüştür ---
                # 'current_page_content' string değil, dictionary listesi olmalı
                page_elements = json.loads(state['current_page_content'])
                
                # agent_id'leri ve seçicileri hızlı erişim için bir haritaya koy
                element_map = {el['agent_id']: el['selector'] for el in page_elements}

                for action in parsed_json.get("actions", []):
                    if "target_agent_id" in action:
                        agent_id = action["target_agent_id"]
                        # Gerçek seçiciyi haritadan bul ve eyleme ekle
                        if agent_id in element_map:
                            action["selector"] = element_map[agent_id]
                        else:
                            print(f"❌ ERROR: LLM returned an invalid agent_id: {agent_id}")
                            action["type"] = "FAIL"
                            action["message"] = f"Could not find element with id {agent_id}."
                
                final_response_payload = parsed_json
                final_response_payload["full_thought_process"] = thought_process

            except json.JSONDecodeError:
                print(f"❌ ERROR: LLM returned a non-JSON response part: {json_str}")
                final_response_payload = {
                    "actions": [{ "type": "FAIL", "explanation": "My reasoning engine produced an invalid format." }],
                    "overall_explanation_of_bundle": "An internal error occurred."
                }
        else:
            print(f"❌ ERROR: LLM response did not contain a <json_response> block.")
            final_response_payload = {
                "actions": [{ "type": "FAIL", "explanation": "My reasoning engine failed to produce a valid action." }],
                "overall_explanation_of_bundle": "An internal error occurred."
            }
            
        # 4. Add the extracted thought process to the final response dictionary
        final_response_payload["full_thought_process"] = thought_process
        
        # Return the complete dictionary to update the state
        return {"final_response": final_response_payload}
    
    def invoke(self, objective: str, page_content: str, previous_actions: List[Dict], user_response: Optional[str]) -> Dict:
        """
        The public method to run a single turn of the agent's reasoning loop.
        """
        # This is the initial input for the graph
        inputs = {
            "objective": objective,
            "current_page_content": page_content,
            "previous_actions": previous_actions,
            "chat_history": [], # chat_history is not used yet, but the state requires it
            "user_response": user_response
        }
        
        # Run the graph from start to finish with the given inputs
        final_state = self.graph.invoke(inputs)
        
        # Return the final response calculated by the last node
        return final_state.get("final_response")