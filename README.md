# JotPilot: The AI Web Automation Agent

JotPilot is an advanced AI agent designed to automate complex tasks on web applications. It combines multimodal perception (vision and HTML analysis) with a sophisticated reasoning engine to understand user objectives and execute them in a live browser environment. Built with LangGraph, JotPilot can create multi-step plans, recover from errors, and interact with web pages in a human-like way.

While developed with Jotform as a primary target, its core architecture is designed to be adaptable to automate any modern web application.

---

## Features

-   **🧠 Intelligent Core:**
    -   **Decision Engine:** The core logic is built on **LangGraph**, enabling a self-correcting loop. The agent can validate its own decisions, learn from failed actions, and create alternative plans to overcome obstacles.
    -   **Multimodal Perception:** The agent uses a hybrid approach to "see" a webpage:
        -   **Vision:** (Optional) Leverages **GPT-4.1-mini** to analyze screenshots, providing a human-like understanding of the visual layout and overcoming limitations of HTML analysis.
        -   **DOM Analysis:** Uses **Playwright** and **BeautifulSoup** to extract and analyze a list of currently visible and interactive elements, providing a structured `VIEW` for the agent.
    -   **Intent Detection:** Intelligently distinguishes between user commands (e.g., "Create a form") and informational questions (e.g., "How do I create a form?"), adapting its behavior accordingly.

-   **📚 RAG Knowledge Base:**
    -   **(Optional) Knowledge Source:** Can be equipped with a knowledge base built from any website. A RAG pipeline (`crawl` -> `chunk` -> `embed`) populates a **Qdrant** vector database.
    -   **Contextual Reasoning:** When enabled, the agent fuses its "sight" (Screenshot/VIEW) with "knowledge" (RAG context) to make more informed decisions.

-   **🔌 Modes of Operation:**
    -   **Stateful API:** A **FastAPI** server (`src/api/server.py`) provides a professional, session-based API. It's designed for a frontend (like a browser extension) to connect to, featuring an `/init` endpoint to start tasks and a `/next_action` endpoint for turn-by-turn interaction.
    -   **Developer Mode:** Includes a `run_developer_mode.py` script for visually testing the agent's actions in a real browser, complete with step-by-step execution, optional user intervention, and detailed debugging logs.

---

## 🛠️ Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Python (3.10+)
-   Docker & Docker Compose

### Step-by-Step Installation

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_PROJECT_URL]
    cd Jotform_agent 
    ```

2.  **Set Up Environment:** It's recommended to use a virtual environment (like venv or Conda).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright Browsers:** This downloads the necessary browser binaries for Playwright.
    ```bash
    playwright install
    ```

5.  **Set Up Your API Key:**
    Copy the `config/env_example.txt` file to a new file named `config/.env`.
    ```bash
    cp config/env_example.txt config/.env
    ```
    Next, open the `config/.env` file and add your OpenAI API key.

6.  **Start the Vector Database:**
    Make sure Docker is running. Then, from the project's root directory, run:
    ```bash
    docker-compose up -d
    ```
    This will start the Qdrant vector database in the background.

---

## Usage

### Phase 1: Build the Knowledge Base (Optional but Recommended)

This one-time setup populates the Qdrant database with knowledge from Jotform's help docs.

a. **Crawl the Data:** `python -m src.crawling.crawler`
b. **Chunk the Data:** `python -m src.chunking.chunker`
c. **Embed and Store the Data:** `python -m src.embedding.embedding_service`

### Phase 2: Run the Agent

You can interact with the project in two main ways:

#### a) Developer Mode (Main Test Environment)

This runs the full agent in a visible browser. It's the primary way to test new objectives and debug the agent's behavior.
```bash
python run_developer_mode.py
```
You can configure the `objective`, `start_url`, `AUTO_MODE`, and other settings directly inside this file.

#### b) API Server (For Frontend Integration)

This starts the FastAPI server, making the agent available to be controlled by a separate application (like a browser extension).
```bash
uvicorn src.api.server:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can see the auto-generated documentation at `http://127.0.0.1:8000/docs`.