# run_developer_mode.py (NİHAİ VE EN GÜNCEL VERSİYON)

import asyncio
import sys
import os
import json
import yaml 
import base64 
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
     # Load config to check if vision is enabled
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    VISION_ENABLED = config.get('features', {}).get('vision_enabled', False)
    print(f"👁️ Vision Mode Enabled: {VISION_ENABLED}")

    # objective = "Create a new ai agent from scratch and name it 'My First Agent'."
    objective = "Create a new form named 'My First form' on Jotform WebSite, and publish it."
    # objective = "Create a new ai agent on Jotform WebSite. Describe it as an algorithm tutor."
    # objective = "Hacettepe yurt sayfasına git ve benim adıma ödeme yap."
    start_url = "https://www.jotform.com/myworkspace/"
    # start_url = "https://barinma.hacettepe.edu.tr/Account/Login?ReturnUrl=%2F"
    
    agent_brain = ActionAgent()

    print("\n--- Configuration Check ---")
    # Agent'ın içindeki OpenAI istemcisinin yeteneğini kontrol et
    model_has_vision = agent_brain.openai_client.has_vision_capability()
    
    if VISION_ENABLED and not model_has_vision:
        print(f"🚨 UYARI: 'vision_enabled: true' ayarlı, ancak seçilen model ('{agent_brain.openai_client.model}') görüş yeteneğine sahip değil.")
        print("   - Lütfen config.yaml dosyasında modeli 'gpt-4o' gibi bir görüş modeliyle güncelleyin.")
        print("   - Hataları önlemek için görüş modu şimdilik kapatılıyor.")
        VISION_ENABLED = False # Hata almamak için görüşü zorla kapat
    elif not VISION_ENABLED and model_has_vision:
        print("ℹ️  Bilgi: Seçili model görüş yeteneğine sahip, ancak 'vision_enabled: false' ayarlı. Sadece metin modu kullanılacak.")
    else:
        print(f"✅ Yapılandırma tutarlı. Görüş Modu: {VISION_ENABLED}")
    print("--------------------------")

    previous_actions = []
    max_turns = 15
    user_response_for_next_turn = None

    AUTO_MODE = True  # Set to True to skip user confirmations

    async with BrowserManager(headless=False) as browser:
        await browser.goto(start_url)
        
        for turn in range(1, max_turns + 1):
            print(f"\n==================== TURN {turn} ====================")

            sleep_time = 1  # seconds
            print(f"⏳ Waiting {sleep_time} seconds for the page to update...")
            await asyncio.sleep(sleep_time)

            # --- 2. SEE & PROCESS ---
            print("👀 Agent is 'seeing' the page and collecting visible elements...")
            visible_elements_html = await browser.get_visible_elements_html()

            # Conditionally take and encode a screenshot
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
                screenshot_base64=screenshot_base64
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
            action_type = first_action.get("type")

            if action_type in ["FINISH", "FAIL"]:
                final_message = first_action.get("status_message")
                print(f"\n🏁 Agent finished or failed: {final_message}")
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

            # --- 6. EXECUTE ACTIONS (The "Translator" Logic) ---
            print("\n🚀 Executing actions...")
            # Bu turda gerçekleşen eylemlerin zengin sonuçlarını tutacak bir liste.
            turn_outcomes_for_history = []
            
            for action in actions_to_take:
                action_type = action.get("type")
                target_index = action.get("target_element_index")
                selector = analyzed_content[target_index].get("selector") if target_index is not None and 0 <= target_index < len(analyzed_content) else None
                
                try:
                    # Debug çıktısını eylemden hemen önce göster
                    print("\n--- DEBUG INFO ---")
                    print(f"Attempting Action: {action_type}")
                    print(f"Target Index: {target_index}")
                    print(f"Resolved Selector: {selector}")
                    print("------------------")

                    if not selector and action_type in ["CLICK", "TYPE"]:
                        raise ValueError(f"Action failed because selector for index {target_index} could not be resolved.")

                    # Eylemi gerçekleştir
                    if action_type == "CLICK":
                        await browser.click(selector)
                    elif action_type == "TYPE":
                        await browser.type(selector, action.get("type_value"))
                    
                    # Eylem başarılı olursa, zengin bir BAŞARI raporu oluştur.
                    turn_outcomes_for_history.append({
                        "action_type": action_type,
                        "description": f"Successfully executed: {action.get('explanation')}"
                    })

                except Exception as e:
                    # Eğer bir hata oluşursa, zengin bir HATA raporu oluştur.
                    print(f"🔥 ACTION FAILED: {e}")
                    turn_outcomes_for_history.append({
                        "action_type": "FAIL",
                        "description": f"Attempted '{action.get('type')}' on index '{target_index}' but it FAILED. Reason: {str(e)}"
                    })
                    # Eylem paketindeki bir adım başarısız olursa, döngüyü kır.
                    break 
            
            # Agent'ın bir sonraki turda göreceği resmi geçmişi, bu turdaki
            # zengin ve gerçekçi sonuçlarla güncelle.
            previous_actions.extend(turn_outcomes_for_history)
            
            sleep_time = 3  # seconds
            print(f"⏳ Waiting {sleep_time} seconds for the page to update...")
            await asyncio.sleep(sleep_time)
            
            previous_actions.extend(actions_to_take)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")