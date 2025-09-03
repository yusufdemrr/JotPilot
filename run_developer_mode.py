# run_developer_mode.py (NİHAİ VE EN GÜNCEL VERSİYON)

import asyncio
import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables at the very top.
load_dotenv('config/.env')

# Add 'src' to the path to allow for clean imports.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from web_interaction.browser_manager import BrowserManager
from agents.action_agent import ActionAgent

async def main():
    """
    The main workflow for running the agent in VISIBLE developer mode.
    This final version has a clean execution loop that trusts the agent's plan.
    """
    # --- 1. SETUP ---
    # objective = "Create a new ai agent from scratch and name it 'My First Agent'."
    objective = "Create a new agent named 'My First agent' on Jotform WebSite. Describe this agent as a algorithm tutor."
    start_url = "https://www.jotform.com/myworkspace/"
    
    agent_brain = ActionAgent()
    previous_actions = []
    max_turns = 15
    user_response_for_next_turn = None

    async with BrowserManager(headless=False) as browser:
        await browser.goto(start_url)
        
        for turn in range(1, max_turns + 1):
            print(f"\n==================== TURN {turn} ====================")

            # --- 2. SEE & PROCESS ---
            print("👀 Agent is 'seeing' the page and collecting visible elements...")
            visible_elements_html = await browser.get_visible_elements_html()

            # --- 3. THINK ---
            print("🧠 Agent is 'thinking' about the next action...")
            final_state = agent_brain.invoke(
                objective=objective,
                visible_elements_html=visible_elements_html,
                previous_actions=previous_actions,
                user_response=user_response_for_next_turn
            )
            user_response_for_next_turn = None

            # --- 4. OBSERVE ---
            response_json = final_state.get("final_response", {})
            analyzed_content = final_state.get("analyzed_content", []) # Get the analysis result

            thought_process = response_json.get("full_thought_process", "No thoughts provided.")
            actions_to_take = response_json.get("actions", [])
            
            print("\n--- Agent's Thought Process ---")
            print(thought_process)
            print("\n--- Agent's Decided Actions ---")
            print(json.dumps(actions_to_take, indent=2))
            
            # --- 5. HANDLE NON-EXECUTABLE & FINAL ACTIONS ---
            if not actions_to_take:
                print("\n🏁 Agent did not decide on an action. Exiting loop.")
                break
            
            first_action = actions_to_take[0]
            if first_action.get("type") in ["FINISH", "FAIL"]:
                print(f"\n🏁 Agent finished or failed: {first_action.get('message')}")
                break
            
            if first_action.get("type") == "ASK_USER":
                question_for_user = first_action.get("question")
                print(f"\n🤔 AGENT ASKS: {question_for_user}")
                user_response_for_next_turn = input("Your response: ")
                previous_actions.extend(actions_to_take)
                continue
            
            user_input = input("\n👉 Press Enter to EXECUTE the actions, or type 'exit' to stop: ")
            if user_input.lower() == 'exit':
                break

            # --- 6. EXECUTE ACTIONS (The "Translator" Logic) ---
            print("\n🚀 Executing actions...")
            for action in actions_to_take:
                action_type = action.get("type")
                target_index = action.get("target_element_index")

                # The script translates the index to a selector using the analysis
                # that the agent conveniently returned to us.
                if target_index is not None and 0 <= target_index < len(analyzed_content):
                    selector = analyzed_content[target_index].get("selector")
                    
                    print("\n--- DEBUG INFO ---")
                    print(f"Action Type: {action_type}")
                    print(f"Target Index: {target_index}")
                    print(f"Resolved Selector: {selector}")
                    print("------------------")

                    if action_type == "CLICK":
                        await browser.click(selector)
                    elif action_type == "TYPE":
                        await browser.type(selector, action.get("value"))
                else:
                    print(f"⚠️ Invalid index ({target_index}) from agent. Skipping action.")
            
            sleep_time = 3  # seconds
            print(f"⏳ Waiting {sleep_time} seconds for the page to update...")
            await asyncio.sleep(sleep_time)
            
            previous_actions.extend(actions_to_take)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")