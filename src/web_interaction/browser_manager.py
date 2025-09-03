# src/web_interaction/browser_manager.py

import asyncio
from typing import Optional, List
from playwright.async_api import async_playwright, Playwright, Browser, Page


class BrowserManager:
    """
    A clean wrapper for Playwright to manage browser interactions asynchronously.
    This class is designed as an asynchronous context manager to ensure
    that browser resources are properly launched and closed.
    
    Usage:
        async with BrowserManager() as browser:
            await browser.goto("https://example.com")
            html = await browser.get_html()
            await browser.click("#some_button")
    """
    
    def __init__(self, headless: bool = True):
        # Store the headless option to be used when launching the browser.
        self.headless = headless
        # Initialize instance variables to None. They will be set in launch().
        # We use lazy initialization to avoid starting Playwright until it's needed.
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def __aenter__(self):
        """
        Asynchronous context manager entry point.
        This method is called when entering an `async with` block.
        It launches the browser and returns the instance of the manager.
        """
        await self.launch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit point.
        This method is called when exiting the `async with` block.
        It ensures that the browser is always closed, even if errors occur.
        """
        await self.close()

    async def launch(self):
        """
        Starts the Playwright instance and launches a new browser page.
        This is the core setup method.
        """
        # Start Playwright
        self.playwright = await async_playwright().start()
        # Launch a Chromium browser instance using the headless option passed during initialization.
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        # Create a new page (tab) in the browser.
        self.page = await self.browser.new_page()
        print("✅ BrowserManager: Playwright started and browser launched.")

    async def close(self):
        """
        Gracefully closes the browser page and the Playwright instance.
        """
        if self.page and not self.page.is_closed():
            await self.page.close()
        if self.browser and self.browser.is_connected():
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("✅ BrowserManager: Browser and Playwright instance closed.")

    async def goto(self, url: str):
        """
        Navigates the browser page to the specified URL.
        
        Args:
            url (str): The URL to navigate to.
        """
        if not self.page:
            raise ConnectionError("Browser is not launched. Call launch() first.")
        
        print(f"🌍 Navigating to {url}...")
        # 'wait_until="domcontentloaded"' waits for the initial HTML document to be loaded and parsed.
        await self.page.goto(url, wait_until="domcontentloaded")
        print(f"👍 Navigated successfully.")

    async def get_html(self) -> str:
        """
        Retrieves the full HTML content of the current page.

        Returns:
            str: The HTML content of the page.
        """
        if not self.page:
            raise ConnectionError("Browser is not launched.")
        
        return await self.page.content()

    async def click(self, selector: str):
            """
            Intelligently clicks an element. It first checks if the element
            is a link that opens in a new tab (`target="_blank"`).
            It handles both new-tab and same-tab navigations robustly.
            """
            if not self.page:
                raise ConnectionError("Browser is not launched.")
                
            print(f"🖱️ Clicking element with selector: {selector}")
            
            try:
                target_element = self.page.locator(selector).first
                await target_element.wait_for(state="visible", timeout=10000)

                # --- YENİ VE AKILLI MANTIK ---
                # Adım 1: Elementin yeni bir sekmede açılıp açılmayacağını kontrol et.
                target_attribute = await target_element.get_attribute('target')

                if target_attribute == '_blank':
                    # --- SENARYO A: YENİ SEKME AÇILACAK ---
                    print("...Element is a link that opens a new tab. Expecting new page...")
                    async with self.page.context.expect_page(timeout=5000) as new_page_info:
                        await target_element.click()
                    
                    new_page = await new_page_info.value
                    self.page = new_page # Fokusumuzu yeni sayfaya geçir
                    print("📄 New page detected. Switched focus.")
                else:
                    # --- SENARYO B: MEVCUT SAYFA DEĞİŞECEK ---
                    print("...Element navigates in the current tab. Clicking normally...")
                    await target_element.click()

                # Her iki senaryodan sonra da, aktif olan sayfanın yüklenmesini bekle.
                await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                print("✅ Click successful and page is ready.")

            except Exception as e:
                print(f"❌ ERROR during click on '{selector}': {e}")
                raise

    async def type(self, selector: str, text: str):
        """
        Types the given text into an element identified by its CSS selector.
        Uses `fill` which is often faster and more reliable than typing character by character.

        Args:
            selector (str): The CSS selector of the input element.
            text (str): The text to type into the element.
        """
        if not self.page:
            raise ConnectionError("Browser is not launched.")

        print(f"⌨️ Typing '{text}' into element: {selector}")
        await self.page.fill(selector, text, timeout=5000)
        print("👍 Typing successful.")

    async def get_visible_elements_html(self) -> List[str]:
        """
        Finds all truly interactable elements on the page, including those
        deeply nested in Shadow DOMs. It first finds all candidates recursively
        and then applies a strict set of visibility filters.
        """
        if not self.page:
            raise ConnectionError("Browser is not launched.")

        js_script = """
        () => {
            const allCandidates = [];
            const selectors = [
                'a[href]', 'button', 'input:not([type=hidden])', 'textarea', 
                'select', '[role=button]', '[role=link]', 'li[tabindex="0"]', 
                'li.field-item', '[onclick]'
            ].join(', ');

            // 1. RECURSIVE SEARCH to find all candidates, even in Shadow DOMs
            function findElements(startElement) {
                // Search in the light DOM of the current element
                startElement.querySelectorAll(selectors).forEach(el => allCandidates.push(el));

                // Search inside all shadow roots within the current element
                startElement.querySelectorAll('*').forEach(el => {
                    if (el.shadowRoot) {
                        findElements(el.shadowRoot); // Recursive call for the shadow root
                    }
                });
            }
            
            // Start the search from the main document body.
            findElements(document.body);

            // 2. STRICT FILTERING on all found candidates
            const finalElementsHTML = [];
            const processedElements = new Set();
            
            for (const el of allCandidates) {
                // If element or its parent was already processed, skip. Prevents seeing a button and its inner text as two separate items.
                if (Array.from(processedElements).some(p => p === el || p.contains(el))) {
                    continue;
                }
                
                try {
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 1 || rect.height < 1) continue;
                    
                    const style = window.getComputedStyle(el);
                    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') continue;
                    if (el.hasAttribute('disabled')) continue;

                    const isInViewport = rect.top >= 0 && rect.left >= 0 && 
                                       rect.bottom <= window.innerHeight && rect.right <= window.innerWidth;
                    if (!isInViewport) continue;
                    
                    const topElement = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
                    if (!topElement || (topElement !== el && !el.contains(topElement))) continue;
                    
                    // If an element passes all checks, it's a valid top-level interactive element.
                    finalElementsHTML.push(el.outerHTML);
                    processedElements.add(el);

                } catch (e) { /* Ignore stale elements */ }
            }
            
            return finalElementsHTML;
        }
        """
        
        print("🕵️  Finding all interactable elements recursively (incl. Shadow DOMs) and filtering...")
        try:
            visible_elements_html = await self.page.evaluate(js_script)
            print(f"👍 Found {len(visible_elements_html)} top-level, visible, and interactable elements.")
            return visible_elements_html
        except Exception as e:
            print(f"❌ ERROR during final element analysis: {e}")
            return []


async def _test_run():
    """A simple test function to demonstrate BrowserManager usage."""
    print("--- Testing BrowserManager ---")
    async with BrowserManager() as browser:
        await browser.goto("https://www.jotform.com/")
        
        # 'Login' yazan ve link olan elementi bulup tıkla
        # Bu CSS selector'ı, metni 'Login' olan bir <a> etiketini hedefler.
        await browser.click("a:has-text('Login')")
        
        print("Login butonuna tıklandı. Yeni sayfanın HTML'i alınıyor...")
        html = await browser.get_html()
        print(f"Login sayfasından {len(html)} karakter içerik alındı.")

    print("--- Test complete ---")

if __name__ == "__main__":
    # To run this test, you first need to install playwright:
    # pip install playwright
    # playwright install
    asyncio.run(_test_run())