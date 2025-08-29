import asyncio
import sys
import os
import json

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from web_interaction.browser_manager import BrowserManager
from web_interaction.page_analyzer import PageAnalyzer
from agents.action_agent import ActionAgent

# run_developer_mode.py içindeki main fonksiyonu

async def main():
    """
    The main workflow for running the agent in VISIBLE developer mode.
    It now handles the ASK_USER action and feeds back the response.
    """
    dotenv_path = 'config/.env'
    load_dotenv(dotenv_path=dotenv_path)
    
    objective = "Create a new ai agent from scratch, name it 'My First agent'."
    start_url = "https://www.jotform.com/workspace/"
    
    agent_brain = ActionAgent()
    page_analyzer = PageAnalyzer()
    previous_actions = []
    max_turns = 15 # Let's give it a bit more room

    # --- YENİ: Kullanıcı cevabını tutacak değişken ---
    user_response_for_next_turn = None

    async with BrowserManager(headless=False) as browser:
        await browser.goto(start_url)
        
        for turn in range(1, max_turns + 1):
            print(f"\n==================== TURN {turn} ====================")

            # --- SEE and PROCESS SIGHT ---        
            print("👀 Agent is 'seeing' the page...")
            raw_html = await browser.get_html()
            simplified_html = page_analyzer.analyze(raw_html)

            # --- THINK ---
            print("🧠 Agent is 'thinking' about the next action...")
            response_json = agent_brain.invoke(
                objective=objective,
                page_content=json.dumps(simplified_html, indent=2),
                previous_actions=previous_actions,
                user_response=user_response_for_next_turn # <-- Cevabı agent'a gönder
            )
            # Bir sonraki tur için cevabı sıfırla
            user_response_for_next_turn = None

            # --- OBSERVE ---
            thought_process = response_json.get("full_thought_process", "No thoughts provided.")
            actions_to_take = response_json.get("actions", [])
            
            print("\n--- Agent's Thought Process ---")
            print(thought_process)
            print("---------------------------------")
            
            print("\n--- Agent's Decided Actions ---")
            print(json.dumps(actions_to_take, indent=2))
            print("---------------------------------")
            
            if not actions_to_take:
                print("\n🏁 Agent did not decide on an action. Exiting loop.")
                break
            
            # --- DECIDE and PREPARE FOR ACTION ---
            first_action = actions_to_take[0]
            action_type = first_action.get("type")

            if action_type in ["FINISH", "FAIL"]:
                print(f"\n🏁 Agent finished or failed: {first_action.get('message')}")
                break
            
            # Agent'ın sorduğu soruyu bize yönlendir
            if action_type == "ASK_USER":
                question_for_user = first_action.get("question")
                print(f"\n🤔 AGENT ASKS: {question_for_user}")
                user_response_for_next_turn = input("Your response: ")
                # Soru sorma eylemini de geçmişe ekle
                previous_actions.extend(actions_to_take)
                continue # Eylem gerçekleştirme adımını atla ve döngünün başına dön
            
            # Diğer eylemler için onay bekle ve gerçekleştir
            user_input = input("\n👉 Press Enter to EXECUTE the actions and continue, or type 'exit' to stop: ")
            if user_input.lower() == 'exit':
                break

            print("\n🚀 Executing actions...")
            for action in actions_to_take:
                action_type = action.get("type")
                selector = action.get("selector")
                value = action.get("value")
                
                if action_type == "CLICK":
                    await browser.click(selector)
                elif action_type == "TYPE":
                    await browser.type(selector, value)
                
                await asyncio.sleep(2)

            # #! --- GÜNCELLENMİŞ EYLEM BLOĞU --- Aynı anda 2 veya daha fazla eylem sorununa çözüm
            # # The script now executes the action(s) in the bundle and then
            # # WILL LOOP BACK to re-see and re-think.
            # print("\n🚀 Executing actions...")
            # for action in actions_to_take:
            #     # Eylemi gerçekleştir
            #     if action.get("type") == "CLICK":
            #         await browser.click(action.get("selector"))
            #     elif action.get("type") == "TYPE":
            #         await browser.type(action.get("selector"), action.get("value"))
                
            #     # Her eylemden sonra kısa bir bekleme süresi, sayfanın oturması için iyidir.
            #     await asyncio.sleep(2)


            previous_actions.extend(actions_to_take)

if __name__ == "__main__":
    # You may need to update BrowserManager's __init__ to accept the headless parameter
    # Go to `src/web_interaction/browser_manager.py` and change:
    # def __init__(self): -> def __init__(self, headless: bool = True):
    # And in the launch() method, change:
    # self.browser = await self.playwright.chromium.launch(headless=True) -> self.browser = await self.playwright.chromium.launch(headless=self.headless)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")