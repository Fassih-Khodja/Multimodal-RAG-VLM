import argparse
import base64
from pathlib import Path

# User mentioned they already installed ollama
import ollama

# Import our previously created search function
from search_qdrant import search_vlm

def get_base64_image(image_path: str) -> str:
    """Read an image file and return its base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_moondream(prompt: str):
    """docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
    Takes a user prompt, retrieves relevant text chunks and images using Qdrant,
    constructs a context-rich prompt, and asks the Ollama moondream model.
    """
    print("\n--- 1. Retrieving context from Qdrant database ---")
    results = search_vlm(prompt=prompt)
    
    # ---------------------------------------------------------
    # 2. Extract Context Text
    # ---------------------------------------------------------
    context_text_blocks = []
    
    # Add top-3 retrieved text chunks
    for i, chunk in enumerate(results["chunks"]):
        text = chunk.get("text", "")
        if text:
            context_text_blocks.append(f"[Retrieved Text Chunk {i+1}]:\n{text}")
            print(f"Added text chunk {text} to context.")
            
    # (Note: In this implementation, the context text of nearby images 
    # is inherently part of the 'chunks' we retrieved above, as well as markdown tables 
    # if they were parsed as text blocks)
    
    context_text = "\n\n".join(context_text_blocks)
    if not context_text:
        context_text = "No relevant text chunks found."
        
    # ---------------------------------------------------------
    # 3. Collect All Unique Images
    # ---------------------------------------------------------
    image_paths = set()
    
    # A. From directly retrieved images
    for img in results["images"]:
        path = img.get("image_path")
        if path:
            image_paths.add(path)
            print(f"Found directly retrieved image: {path}")
            
    # B. From nearby_image_ids in retrieved text chunks
    for chunk in results["chunks"]:
        nearby = chunk.get("nearby_images", [])
        for img_id in nearby:
            # Using the standard image path convention
            expected_path = f"extracted/images/{img_id}.png"
            if Path(expected_path).exists():
                image_paths.add(expected_path)
    
    # Convert images to base64 for ollama
    base64_images = []
    for ipath in image_paths:
        try:
            base64_images.append(get_base64_image(ipath))
        except Exception as e:
            print(f"Warning: Could not read image {ipath}: {e}")
            
    # ---------------------------------------------------------
    # 4. Construct the Final Prompt
    # ---------------------------------------------------------
    system_prompt = "You are a technical manual assistant. Answer only based on the provided context."
    
    full_prompt = f"""System: {system_prompt}

Context Text:
{context_text}

Images: [Attached {len(base64_images)} image(s) to the message payload]

User Question: {prompt}"""
    
    # ---------------------------------------------------------
    # 5. Call Ollama (Moondream)
    # ---------------------------------------------------------
    print(f"\n--- 2. Sending {len(base64_images)} image(s) and context to Moondream ---")
    
    response = ollama.chat(
        model='llava-phi3',
        messages=[
            {
                'role': 'user',
                'content': full_prompt,
                'images': base64_images  # Ollama natively takes the base64 lists here
            }
        ]
    )
    
    return response['message']['content']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Moondream using VLM RAG")
    parser.add_argument("--prompt", type=str, required=True, help="Your question")
    args = parser.parse_args()
    
    answer = ask_moondream(prompt=args.prompt)
    print("\n==================== ANSWER ====================")
    print(answer)
    print("================================================\n")
