# Jotform AI RAG Agent

This project is a RAG (Retrieval-Augmented Generation) based AI agent, built with LangGraph, that answers user questions about Jotform using its help documentation as a knowledge base.

---

## 🚀 Features

-   **Data Collection:** Automatically scrapes data from Jotform's help pages using `crawler.py`.
-   **Data Processing:** intelligently splits the raw text into semantically meaningful chunks optimized for RAG using `chunker.py`.
-   **Vector Database:** Converts text chunks into vector embeddings and stores them in a **Qdrant** database using `embedding_service.py`.
-   **Intelligent Chat:** Generates context-aware answers to user questions by retrieving relevant information from Qdrant and feeding it to an LLM.
-   **Scalable Architecture:** The entire chat flow is built on **LangGraph**, allowing for easy integration of new tools and capabilities in the future.
-   **Memory:** Remembers the conversation history to provide a more natural and coherent chat experience.

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

Once the installation is complete, you can run the agent. The process has two main phases.

### Phase 1: Build the Knowledge Base (One-time setup)

These steps collect the documentation, process it, and load it into the agent's memory (Qdrant).

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

### Phase 2: Run the Chatbot

After building the knowledge base, you can start interacting with the agent.
```bash
python -m src.llm.chatbot_langgraph
```
You will see a prompt in your terminal. Start asking your questions about Jotform! Type `exit` to quit the session.