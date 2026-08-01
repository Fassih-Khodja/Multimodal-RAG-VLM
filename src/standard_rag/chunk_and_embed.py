from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _extend_unique(target: List[str], items: Iterable[str]) -> None:
    seen = set(target)
    for item in items:
        if item not in seen:
            target.append(item)
            seen.add(item)


def _split_long_text(text: str, max_len: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    # Prefer sentence ends followed by newlines (paragraph boundaries).
    parts = re.split(r"(?<=[.!?])\s*\n+", text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 1:
        # Fallback: sentence ends followed by whitespace.
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 1:
        return [text]

    segments: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}\n{part}"
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                segments.append(current)
                current = part
            else:
                segments.append(part)
                current = ""
    if current:
        segments.append(current)

    return segments


def _make_pieces(
    text_blocks: List[Dict[str, Any]],
    max_len: int,
) -> List[Dict[str, Any]]:
    pieces: List[Dict[str, Any]] = []
    for block in text_blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        block_id = str(block.get("block_id", ""))
        nearby_images = block.get("nearby_images", [])
        if len(text) > max_len:
            for part in _split_long_text(text, max_len):
                if not part:
                    continue
                pieces.append(
                    {
                        "text": part,
                        "block_ids": [block_id],
                        "nearby_images": list(nearby_images),
                    }
                )
        else:
            pieces.append(
                {
                    "text": text,
                    "block_ids": [block_id],
                    "nearby_images": list(nearby_images),
                }
            )
    return pieces


def build_chunks(
    text_blocks: List[Dict[str, Any]],
    min_len: int = 100,
    max_len: int = 500,
) -> List[Dict[str, Any]]:
    pieces = _make_pieces(text_blocks, max_len=max_len)
    chunks: List[Dict[str, Any]] = []

    idx = 0
    chunk_index = 0
    while idx < len(pieces):
        block_ids: List[str] = []
        nearby_images: List[str] = []
        text_parts: List[str] = []
        length = 0

        while idx < len(pieces) and (length < min_len or not text_parts):
            piece = pieces[idx]
            text_parts.append(piece["text"])
            length += len(piece["text"]) + (1 if len(text_parts) > 1 else 0)
            _extend_unique(block_ids, piece["block_ids"])
            _extend_unique(nearby_images, piece["nearby_images"])
            idx += 1

        chunk_text = "\n".join(text_parts).strip()
        if not chunk_text:
            continue

        sub_texts = (
            _split_long_text(chunk_text, max_len) if len(chunk_text) > max_len else [chunk_text]
        )
        for sub_text in sub_texts:
            if not sub_text:
                continue
            chunk_index += 1
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_index:06d}",
                    "block_ids": block_ids,
                    "text": sub_text,
                    "nearby_images": nearby_images,
                }
            )

    return chunks


def _pick_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str,
    device: str,
) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install it with: pip install sentence-transformers"
        ) from exc

    if not chunks:
        return

    model = SentenceTransformer(model_name, device=device)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()


def embed_images(
    images: List[Dict[str, Any]],
    model_name: str,
    device: str,
    base_dir: Path,
) -> List[Dict[str, Any]]:
    def _to_vector(features: Any) -> List[float]:
        import torch

        if isinstance(features, torch.Tensor):
            tensor = features
        elif hasattr(features, "image_embeds"):
            tensor = features.image_embeds
        elif hasattr(features, "pooler_output"):
            tensor = features.pooler_output
        else:
            raise TypeError("Unsupported image features output type")

        return tensor.detach().cpu().numpy().reshape(-1).tolist()

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers, torch, and pillow are required. Install with: pip install transformers torch Pillow"
        ) from exc

    if not images:
        return []

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()

    results: List[Dict[str, Any]] = []
    for record in images:
        image_path = Path(str(record.get("image_path", "")))
        if not image_path.is_absolute():
            candidate = base_dir / image_path
            if candidate.exists():
                image_path = candidate
            else:
                candidate = base_dir.parent / image_path
                if candidate.exists():
                    image_path = candidate
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        embedding = _to_vector(features)

        output = dict(record)
        output["image_path"] = str(image_path)
        output["embedding"] = embedding
        results.append(output)

    return results


def build_chunks_and_embeddings(
    texts_json_path: str,
    images_json_path: str,
    output_dir: str | None = None,
    min_len: int = 100,
    max_len: int = 500,
    text_model: str = "all-MiniLM-L6-v2",
    image_model: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> Dict[str, Path]:
    texts_path = Path(texts_json_path)
    images_path = Path(images_json_path)
    if not texts_path.exists():
        raise FileNotFoundError(f"texts.json not found: {texts_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"images.json not found: {images_path}")

    out_dir = Path(output_dir) if output_dir else texts_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    text_blocks = _read_json(texts_path)
    images = _read_json(images_path)

    chunks = build_chunks(text_blocks, min_len=min_len, max_len=max_len)

    device_name = _pick_device(device)
    embed_chunks(chunks, model_name=text_model, device=device_name)
    image_embeddings = embed_images(
        images,
        model_name=image_model,
        device=device_name,
        base_dir=images_path.parent,
    )

    chunks_path = out_dir / "chunks.json"
    images_out_path = out_dir / "image_embeddings.json"

    _write_json(chunks_path, chunks)
    _write_json(images_out_path, image_embeddings)

    return {
        "chunks_json": chunks_path,
        "image_embeddings_json": images_out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk texts and compute text/image embeddings."
    )
    parser.add_argument("--texts-json", required=True, help="Path to texts.json")
    parser.add_argument("--images-json", required=True, help="Path to images.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for chunks.json and image_embeddings.json",
    )
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument(
        "--text-model", default="all-MiniLM-L6-v2", help="Sentence-transformers model"
    )
    parser.add_argument(
        "--image-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (cpu, cuda). Defaults to auto.",
    )
    args = parser.parse_args()

    build_chunks_and_embeddings(
        texts_json_path=args.texts_json,
        images_json_path=args.images_json,
        output_dir=args.output_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        text_model=args.text_model,
        image_model=args.image_model,
        device=args.device,
    )
