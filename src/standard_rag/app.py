import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import from existing scripts
from pdf_extract import extract_pdf_assets
from chunk_and_embed import build_chunks_and_embeddings
from qdrant_ingest import ingest_to_qdrant
from ask_vlm import ask_openrouter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process")
async def process_pdf_and_prompt(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    pdf_path = f"temp_{file.filename}"
    try:
        # Save uploaded PDF to a temporary file
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print("1. Extracting PDF assets...")
        extracted_paths = extract_pdf_assets(pdf_path, output_dir="extracted")
        
        print("2. Chunking and Embedding...")
        embed_paths = build_chunks_and_embeddings(
            texts_json_path=str(extracted_paths["texts_json"]),
            images_json_path=str(extracted_paths["images_json"]),
            output_dir="extracted",
            text_model="all-MiniLM-L6-v2",
            image_model="openai/clip-vit-base-patch32",
            device=None # auto detect
        )
        
        print("3. Ingesting to Qdrant...")
        ingest_to_qdrant(
            chunks_json=embed_paths["chunks_json"],
            images_json=embed_paths["image_embeddings_json"],
            url="http://localhost:6333",
            text_collection="vlm_text_chunks",
            image_collection="vlm_images",
            batch_size=64,
            recreate=True  # Clear and recreate to answer context purely based on this upload
        )
        
        print("4. Asking VLM...")
        answer = ask_openrouter(prompt)
        
        return {"status": "success", "answer": answer}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
