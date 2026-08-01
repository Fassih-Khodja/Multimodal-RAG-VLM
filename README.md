# Advanced Multimodal & Agentic RAG System

Welcome to the **Advanced Multimodal & Agentic RAG System** project! This repository contains a comprehensive suite of AI tools demonstrating state-of-the-art Retrieval-Augmented Generation (RAG) capabilities using Vision-Language Models (VLMs) and Agentic AI frameworks.

This project was built to explore how we can bridge the gap between text-only language models and visually rich documents (like manuals, diagrams, and blueprints), and how we can empower these models with reasoning and tool-use via Agentic AI.

## 🚀 Two Main Components

This repository is divided into two major projects that build upon each other:

### 1. Multimodal RAG System (`src/standard_rag`)
A complete pipeline to extract, embed, and query both text and images from complex PDFs. 
*   **Data Extraction**: Parses text, images, and tables from PDFs.
*   **Multimodal Embedding**: Uses CLIP and text embedding models to map visuals and text into a shared vector space.
*   **Vector Database**: Utilizes **Qdrant** to store and retrieve these embeddings efficiently.
*   **VLM Integration**: Uses **OpenRouter** to query advanced Vision-Language Models (like LLaVA or Qwen-VL) passing both the user's prompt and retrieved visual context.
*   **Interactive UI**: Includes a FastAPI web interface to upload PDFs, process them, and chat with your documents visually.

### 2. Agentic AI RAG (`src/agentic_rag`)
Enhances the RAG pipeline with Agentic behavior, allowing the AI to reason, use tools, and make decisions to answer complex queries.
*   **ReAct Architecture**: Implements the Reason-Act framework, giving the LLM the ability to break down problems into steps and execute tools.
*   **LangGraph**: Uses LangGraph to orchestrate complex, multi-agent workflows and stateful cyclic processes for deeper document analysis.
*   **Tool Use**: Agents can independently decide when to search the vector database, process an image, or calculate a metric based on the user's query.

---

## 📂 Project Structure

```
.
├── src/
│   ├── standard_rag/       # Core multimodal RAG extraction, ingestion, and UI scripts
│   └── agentic_rag/        # Advanced LangGraph and ReAct agent implementations
├── tutorials/              # Jupyter notebooks explaining core concepts step-by-step
├── Project_Explanation.md  # Deep-dive architecture and design decisions
└── requirements.txt        # Python dependencies
```

## 🛠️ Technologies & Skills Demonstrated
- **AI / LLMs**: Vision-Language Models (VLMs), OpenRouter API, CLIP Embeddings.
- **Frameworks**: LangGraph, LangChain, FastAPI.
- **Agentic AI**: ReAct Pattern, Stateful Agents, Tool calling.
- **Vector Search**: Qdrant, Multimodal Retrieval.
- **Programming**: Python, Object-Oriented Design, API Integration.

## ⚙️ How to Run Locally

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Export your OpenRouter API key to use the VLM:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### 3. Run Qdrant Database (Docker)
```bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Interactive Web UI (Standard RAG)
You can run the web interface to upload a document and test the multimodal RAG:
```bash
cd src/standard_rag
python app.py
```
Then visit `http://localhost:8000` in your browser.

### 5. Running Agentic AI Scripts
Navigate to the agentic directory to test the LangGraph and ReAct agents:
```bash
cd src/agentic_rag
python langgraph_agent.py
```

## 📚 Learning Resources
If you are new to the concepts used in this project, check out the `tutorials/` directory. It contains interactive notebooks (`vlm_rag_basics.ipynb` and `agentic_ai_basics.ipynb`) designed to gently introduce the theory and code behind Multimodal RAGs and AI Agents.

---
*Created as an End-of-Studies (PFE) demonstration project to showcase advanced modern AI integration.*
