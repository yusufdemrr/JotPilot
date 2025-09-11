# run_developer_mode.py

import asyncio
import sys
import os
import json
import yaml 
import base64 
from dotenv import load_dotenv

# --- For asynchronous, non-blocking user input ---
import selectors
selector = selectors.DefaultSelector()
loop = asyncio.get_event_loop()
user_input_buffer = ""

def on_user_input():
    global user_input_buffer
    user_input_buffer = sys.stdin.readline().strip()
# ---------------------------------------------------

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
    global user_input_buffer # To hold user input between async calls
    
    # --- 1. SETUP ---
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    VISION_ENABLED = config.get('features', {}).get('vision_enabled', False)
    print(f"👁️ Vision Mode Enabled: {VISION_ENABLED}")

    # --- Define the main objective for the agent ---
    # objective = "Create a new form, add a few basic elements, and publish."
    objective = "Who is the CEO of Jotform?"
    # objective = "Create a course registration form suitable for any school or institution."
    # objective = "Ekşi sözlüğe sıfırdan kayıt ol sonasında bir entry gir istediğin konuda."
    # objective = "Create a new ai agent on Jotform WebSite. Describe it as an algorithm tutor."
    # objective = "Create a new form on Jotform WebSite. Create a heading and type 'Hello World' in it and publish."
    # objective = "Create a new app on Jotform WebSite. Create a text and type 'Hello World' in it and publish."
    # objective = "Hacettepe yurt sayfasına git ve benim adıma ödeme yap."
    # objective = "Arabam.com sitesinde çakal kasa bmw ilanlarını bul. Fiyata göre sırala"
    start_url = "https://www.jotform.com/myworkspace/"
    # start_url = "https://eksisozluk.com/"
    # start_url = "https://barinma.hacettepe.edu.tr/Account/Login?ReturnUrl=%2F"
    # start_url = "https://www.arabam.com/"
    
    agent_brain = ActionAgent()

    print("\n--- Configuration Check ---")
    # Check the capability of the OpenAI client within the agent
    model_has_vision = agent_brain.openai_client.has_vision_capability()
    
    if VISION_ENABLED and not model_has_vision:
        print(f"🚨 WARNING: 'vision_enabled: true' is set, but the selected model ('{agent_brain.openai_client.model}') does not have vision capability.")
        print("   - Please update the model in config.yaml to a vision model like 'gpt-4o'.")
        print("   - To prevent errors, vision mode is being temporarily disabled.")
        VISION_ENABLED = False # Forcibly disable vision to avoid errors
    elif not VISION_ENABLED and model_has_vision:
        print("ℹ️ Info: The selected model has vision capability, but 'vision_enabled: false' is set. Only text mode will be used.")
    else:
        print(f"✅ Configuration is consistent. Vision Mode: {VISION_ENABLED}")
    print("--------------------------")

    previous_actions = []
    max_turns = 15                             #* Maximum number of turns to prevent infinite loops 
    last_analyzed_content_for_next_turn = None # Stores the last analysis result to provide context for the next turn.

    # --- Start the non-blocking input listener ---
    try:
        loop.add_reader(sys.stdin.fileno(), on_user_input)
        print("🎤 User intervention is active. Type a command and press Enter at any time.")
    except Exception as e:
        print(f"⚠️ Could not start non-blocking input reader (may not work in all terminals): {e}")

    AUTO_MODE = True
    user_response_for_next_turn = None

    async with BrowserManager(headless=False) as browser:
        await browser.goto(start_url)
        
        for turn in range(1, max_turns + 1):
            print(f"\n==================== TURN {turn} ====================")

            if user_input_buffer:
                print(f"🙋 User intervention received: '{user_input_buffer}'")
                user_response_for_next_turn = user_input_buffer
                user_input_buffer = "" # Clear the buffer

            sleep_time = 1
            print(f"⏳ Waiting {sleep_time} seconds for the page to update...")
            await asyncio.sleep(sleep_time)

            # --- 2. SEE & PROCESS ---
            print("👀 Agent is 'seeing' the page and collecting visible elements...")
            visible_elements_html = await browser.get_visible_elements_html()

            screenshot_base64 = None
            if VISION_ENABLED:
                print("📸 Taking a screenshot for visual analysis...")
                screenshot_bytes = await browser.page.screenshot()
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode()
                
            # --- 3. THINK ---
            print("🧠 Agent is 'thinking' about the next action...")
            final_state = agent_brain.invoke(
                objective=objective,
                visible_elements_html=visible_elements_html,
                previous_actions=previous_actions,
                user_response=user_response_for_next_turn,
                screenshot_base64=screenshot_base64,
                last_analyzed_content=last_analyzed_content_for_next_turn
            )
            user_response_for_next_turn = None # Reset after use

            # --- 4. OBSERVE ---
            response_json = final_state.get("final_response", {})
            analyzed_content = final_state.get("analyzed_content", [])
            last_analyzed_content_for_next_turn = analyzed_content

            thought_process = response_json.get("full_thought_process", "No thoughts provided.")
            actions_to_take = response_json.get("actions", [])
            page_summary = response_json.get("page_summary", "Agent did not provide a page summary.")

            print("\n--- Agent's Page Summary ---")
            print(page_summary)
            
            print("\n--- Agent's Thought Process ---")
            print(thought_process)
            print("\n--- Agent's Decided Actions ---")
            print(json.dumps(actions_to_take, indent=2))
            
            # --- 5. HANDLE NON-EXECUTABLE & FINAL ACTIONS ---
            if not actions_to_take:
                print("\n🏁 Agent did not decide on an action. Exiting loop.")
                break
            
            first_action = actions_to_take[0]
            action_type = first_action.get("type")

            if action_type in ["FINISH", "FAIL"]:
                final_message = first_action.get("status_message")
                print(f"\n🏁 Agent finished or failed: {final_message}")
                sleep_time = 3
                print(f"⏳ Waiting {sleep_time} seconds before closing...")
                await asyncio.sleep(sleep_time)
                break
            
            if action_type == "ASK_USER":
                question_for_user = first_action.get("user_question")
                print(f"\n🤔 AGENT ASKS: {question_for_user}")
                user_response_for_next_turn = input("Your response: ")
                previous_actions.extend(actions_to_take)
                continue
            
            #* Ask for user confirmation before executing actions
            if not AUTO_MODE:
                user_input = input("\n👉 Press Enter to EXECUTE the actions, or type 'exit' to stop: ")
                if user_input.lower() == 'exit':
                    break

            # --- 6. EXECUTE ACTIONS ---
            print("\n🚀 Executing actions...")
            turn_outcomes_for_history = []
            
            for action in actions_to_take:
                try:
                    action_type = action.get("type")
                    target_index = action.get("target_element_index")
                    
                    target_element_data = analyzed_content[target_index] if target_index is not None and 0 <= target_index < len(analyzed_content) else None

                    if target_element_data is None and action_type in ["CLICK", "TYPE"]:
                        raise ValueError(f"Agent chose an invalid index: {target_index}")
                    
                    selector = target_element_data.get("selector")
                    tag = target_element_data.get("tag")
                    value = action.get("type_value")
                    
                    print("\n--- DEBUG INFO ---")
                    print(f"Attempting Action: {action_type}")
                    print(f"Target Index: {target_index}")
                    print(f"Resolved Selector: {selector}")
                    print("------------------")

                    if not selector and action_type in ["CLICK", "TYPE"]:
                        raise ValueError(f"Action failed because selector for index {target_index} could not be resolved.")

                    if action_type == "CLICK":
                        await browser.click(selector)
                    elif action_type == "TYPE":
                        if tag in ["input", "textarea"]:
                            await browser.fill_text(selector, value)
                        else:
                            await browser.click_and_type(selector, value)
                        
                    turn_outcomes_for_history.append({
                        "action_type": action_type,
                        "description": f"Successfully executed: {action.get('explanation')}"
                    })

                except Exception as e:
                    print(f"🔥 ACTION FAILED: {e}")
                    turn_outcomes_for_history.append({
                        "action_type": "FAIL",
                        "description": f"Attempted '{action.get('type')}' on index '{target_index}' but it FAILED. Reason: {str(e)}"
                    })
                    break 
            
            # Update the official history with the rich and realistic outcomes from this turn.
            previous_actions.extend(turn_outcomes_for_history)
            
            sleep_time = 3
            print(f"⏳ Waiting {sleep_time} seconds for the page to update...")
            await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")