import argparse
import os
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image

# Import OpenAI (used for OpenRouter)
from openai import OpenAI

# Import our previously created search function
from search_qdrant import search_vlm

# ==========================================
# CONFIGURATION: OPENROUTER API
# ==========================================
# Place your API key here or set it as an environment variable
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# You can change this to another model supported by OpenRouter (e.g. openai/gpt-4o)
MODEL_NAME = "openai/gpt-4o"
# ==========================================

def get_base64_image(image_path: str, max_size=(800, 800)) -> str:
    """Read an image file, resize it, and return the base64 encoded string."""
    img = Image.open(image_path)
    # Convert to RGB if it's not (e.g. RGBA or P)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if larger than max_size while maintaining aspect ratio
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert PIL Image to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def ask_openrouter(prompt: str):
    """docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
    Takes a user prompt, retrieves relevant text chunks and images using Qdrant,
    constructs a context-rich prompt, and asks the model via OpenRouter.
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
    
    # Load images as base64 for OpenAI API
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
    
    # Text part of the message
    user_text = f"""Context Text:
{context_text}

Images: [Attached {len(base64_images)} image(s)]

User Question: {prompt}"""

    # Build the message content (text + images)
    content = [{"type": "text", "text": user_text}]
    for b64_img in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_img}"
            }
        })

    # ---------------------------------------------------------
    # 5. Call OpenRouter
    # ---------------------------------------------------------
    print(f"\n--- 2. Sending {len(base64_images)} image(s) and context to OpenRouter ---")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        max_tokens=1000  # Specify limit to prevent 402 insufficient token credit errors
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask OpenRouter using VLM RAG")
    parser.add_argument("--prompt", type=str, required=True, help="Your question")
    args = parser.parse_args()
    
    answer = ask_openrouter(prompt=args.prompt)
    print("\n==================== ANSWER ====================")
    print(answer)
    print("================================================\n")
