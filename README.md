# Multimodal RAG using Vision-Language Models (VLM)

Welcome to the Multimodal RAG project! This project demonstrates how to build a **Visual RAG (Retrieval-Augmented Generation)** system. Normal RAG systems only process text, completely ignoring crucial visual data like diagrams, charts, and blueprints. This project bridges that gap by extracting both text and images from complex PDFs, embedding them into a unified multimodal vector database, and utilizing an advanced Vision-Language Model to answer your questions accurately based on both text and visual context.

##  Project Structure & Explanation

For a deep dive into how the architecture is designed and the logic behind each phase of the project, please read the **[Project Explanation](Project_Explanation.md)**. It contains a detailed breakdown and architectural diagrams to help you fully grasp the pipeline.

**Are you a beginner?**
If you are new to concepts like RAG, CLIP, Vector Databases, or VLMs, we highly recommend checking out our interactive notebook: **[Learning.ipynb](Learning.ipynb)**. It breaks down these complex concepts into simple, easy-to-understand explanations with practical examples!

##  Why OpenRouter?

Initially, we tested running this pipeline entirely locally using lightweight models like **Moondream** (which is great for CPU environments). However, we found its text generation capabilities to be lacking for complex technical queries. On the other hand, larger models like **LLaVA** or **Qwen-VL** are quite heavy and require significant GPU resources. 

To strike the perfect balance between high-quality responses and performance, we migrated to the **OpenRouter API**, allowing us to utilize powerful, state-of-the-art vision models without the hardware overhead.

##  How to Run the Project

### 1. Setup the Environment

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

We use OpenRouter to access the VLM. You must set your API key as an environment variable (Do not hardcode it in the source files!):

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### 3. Start Qdrant (Vector Database)

We use Qdrant to store our embeddings. Start it locally using Docker:

```bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Extract PDF Assets

Extract text, images, and tables from your technical PDF:

```bash
python pdf_extract.py /path/to/your.pdf --output-dir extracted
```

This will generate parsed JSONs and image files inside the `extracted/` folder.

### 5. Build Chunks and Embeddings

Process the extracted data and generate multimodal embeddings (using CLIP and text embedders):

```bash
python chunk_and_embed.py \
  --texts-json extracted/texts.json \
  --images-json extracted/images.json \
  --output-dir extracted
```

### 6. Ingest into Qdrant

Load the generated chunks and embeddings into your running Qdrant database:

```bash
python qdrant_ingest.py \
  --chunks-json extracted/chunks.json \
  --image-embeddings-json extracted/image_embeddings.json
```

### 7. Ask the VLM

Finally, query your system! The script will retrieve relevant text and images from Qdrant, construct a multimodal prompt, and ask the VLM via OpenRouter:

```bash
python ask_vlm.py --prompt "Based on the wiring diagram, what happens if I cut the red wire?"
```
