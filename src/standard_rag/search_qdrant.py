import argparse
import json
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def search_vlm(prompt: str):
    """
    Takes a text prompt.
    Embeds the text prompt using all-MiniLM-L6-v2 for chunk search,
    and openai/clip-vit-base-patch32 for image search.
    Queries Qdrant and returns the top 3 text chunks and top 2 images.
    """
    # Initialize Qdrant Client (assuming localhost:6333)
    client = QdrantClient(url="http://localhost:6333")
    
    # 1. Embed text using sentence-transformers for text chunks
    print(f"Embedding prompt using all-MiniLM-L6-v2...")
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text_embedding = text_model.encode(prompt).tolist()
    
    # 2. Embed text using CLIP for images
    print(f"Embedding prompt using openai/clip-vit-base-patch32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    # Get the text features (embeddings) and convert to list
    features = clip_model.get_text_features(**inputs)
    
    # Handle both Tensor and BaseModelOutputWithPooling return types
    if hasattr(features, "detach"):
        tensor = features
    elif hasattr(features, "text_embeds"):
        tensor = features.text_embeds
    elif hasattr(features, "pooler_output"):
        tensor = features.pooler_output
    else:
        tensor = features[0]
        
    clip_embedding = tensor.detach().numpy().flatten().tolist()
    
    # 3. Search in Qdrant (Text Chunks)
    print("Searching vlm_text_chunks...")
    text_results = client.query_points(
        collection_name="vlm_text_chunks",
        query=text_embedding,
        limit=3
    )
    
    # 4. Search in Qdrant (Images)
    print("Searching vlm_images...")
    image_results = client.query_points(
        collection_name="vlm_images",
        query=clip_embedding,
        limit=2
    )
    
    # 5. Process and collect the results
    retrieved_chunks = []
    for res in text_results.points:
        payload = res.payload or {}
        retrieved_chunks.append({
            "text": payload.get("text", ""),
            "nearby_images": payload.get("nearby_images", []),
            "block_ids": payload.get("block_ids", []),
            "score": res.score
        })
        
    retrieved_images = []
    for res in image_results.points:
        payload = res.payload or {}
        retrieved_images.append({
            "image_path": payload.get("image_path", ""),
            "page_number": payload.get("page_number", None),
            "nearest_blocks": payload.get("nearest_blocks", []),
            "score": res.score
        })
        
    return {
        "chunks": retrieved_chunks,
        "images": retrieved_images
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search VLM Qdrant database")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to search")
    
    args = parser.parse_args()
    
    results = search_vlm(prompt=args.prompt)
    
    print("\n--- SEARCH RESULTS ---\n")
    print(json.dumps(results, indent=4, ensure_ascii=False))
