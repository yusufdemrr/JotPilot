# Jotform AI RAG Agent

This project is an AI agent designed to interact with the Jotform website. It combines a **RAG (Retrieval-Augmented Generation)** knowledge base with a **web automation agent** built on LangGraph to understand user objectives and execute tasks in a browser.

---

## 🚀 Features

-   **RAG Knowledge Base:**
    -   **Data Collection:** Automatically scrapes data from Jotform's help pages.
    -   **Data Processing:** Intelligently splits raw text into semantically meaningful chunks.
    -   **Vector Database:** Converts text chunks into vector embeddings and stores them in a **Qdrant** database.
-   **Action Agent Core:**
    -   **Web Perception:** Uses `Playwright` to see and `BeautifulSoup` to analyze the content of live web pages.
    -   **Decision Engine:** The core logic is built on **LangGraph**, allowing the agent to create multi-step plans, reason about its actions, and even self-correct on errors.
    -   **Tool Use:** The RAG system functions as a `rag_tool` that the main agent can consult when it needs theoretical knowledge.
-   **Modes of Operation:**
    -   **Stateful API:** A FastAPI server (`src/api/server.py`) provides a session-based, turn-by-turn API for frontend integration.
    -   **Developer Mode:** Includes a `run_developer_mode.py` script to visually test the agent's actions in a real browser window.

---

## 🛠️ Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Python (3.10+)
-   Docker & Docker Compose
-   Conda

### Step-by-Step Installation

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_PROJECT_URL]
    cd Jotform_agent
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    # Create a new conda environment named "jotform_agent" with Python 3.10
    conda create --name jotform_agent python=3.10

    # Activate the environment
    conda activate jotform_agent
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key:**
    Copy the `env_example.txt` file to a new file named `.env`.
    ```bash
    cp env_example.txt .env
    ```
    Next, open the `.env` file with a text editor and add your OpenAI API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```

5.  **Start the Vector Database:**
    Make sure Docker is running. Then, from the project's root directory, run the following command:
    ```bash
    docker-compose up -d
    ```
    This will start the Qdrant vector database in the background.

---

## 🚀 Usage

The project has two main phases: building the knowledge base and running the agent.

### Phase 1: Build the Knowledge Base (One-time setup)

These steps collect the documentation, process it, and load it into the agent's memory (Qdrant). This only needs to be done once or when you want to update the knowledge base.

a. **Crawl the Data:**
```bash
python -m src.crawling.crawler
```

b. **Chunk the Data:**
```bash
python -m src.chunking.chunker
```

c. **Embed and Store the Data:**
```bash
python -m src.embedding.embedding_service
```

### Phase 2: Run the Agent

You can run the agent in two different modes:

#### a) Action Agent (Developer Mode) - *Main Mode*

This is the primary way to run the project. It will open a browser window and execute tasks based on the objective defined in the script.

```bash
python run_developer_mode.py
```
You will see the agent's thought process in the terminal and its actions in the browser window. Press `Enter` to proceed with each step.

#### b) RAG Chatbot (Demo Mode)

This runs the original, simple RAG chatbot for question-answering only, without any web automation.

```bash
python -m src.llm.chatbot