from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_vector(raw: Any) -> List[float] | None:
    if not isinstance(raw, list) or not raw:
        return None

    def _unwrap_singletons(value: Any) -> Any:
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        return value

    vector = _unwrap_singletons(raw)
    if not isinstance(vector, list) or not vector:
        return None

    if all(isinstance(value, (int, float)) for value in vector):
        return [float(value) for value in vector]

    if all(isinstance(value, list) for value in vector):
        candidates = [_unwrap_singletons(value) for value in vector]
        if not candidates:
            return None
        if not all(
            isinstance(value, list)
            and value
            and all(isinstance(item, (int, float)) for item in value)
            for value in candidates
        ):
            return None

        if len(candidates) == 1:
            return [float(item) for item in candidates[0]]

        size = len(candidates[0])
        if not all(len(value) == size for value in candidates):
            raise ValueError("Embedding vectors have inconsistent lengths; cannot pool.")

        # Mean-pool multiple vectors into one.
        totals = [0.0] * size
        for value in candidates:
            for index, item in enumerate(value):
                totals[index] += float(item)
        count = len(candidates)
        return [total / count for total in totals]

    return None


def _first_vector(records: Iterable[Dict[str, Any]], key: str) -> List[float] | None:
    for record in records:
        vector = _normalize_vector(record.get(key))
        if vector:
            return vector
    return None


def _split_embedding(record: Dict[str, Any], key: str) -> Tuple[List[float] | None, Dict[str, Any]]:
    payload = dict(record)
    raw = payload.pop(key, None)
    vector = _normalize_vector(raw)
    return vector, payload


def _point_id(prefix: str, record: Dict[str, Any], index: int) -> str:
    raw = record.get("chunk_id") or record.get("image_id") or str(index)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}:{raw}"))


def _ensure_collection(client, name: str, vector_size: int, recreate: bool) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        if hasattr(vectors, "size") and vectors.size != vector_size:
            if recreate:
                client.delete_collection(name)
            else:
                raise ValueError(
                    f"Collection '{name}' has size {vectors.size}, expected {vector_size}."
                )
        elif not hasattr(vectors, "size"):
            if recreate:
                client.delete_collection(name)
            else:
                raise ValueError(
                    f"Collection '{name}' uses named vectors; cannot verify size."
                )
        else:
            return

    from qdrant_client.models import Distance, VectorParams

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _upsert_records(
    client,
    collection: str,
    records: List[Dict[str, Any]],
    embedding_key: str,
    id_prefix: str,
    batch_size: int,
) -> int:
    from qdrant_client.models import PointStruct

    total = 0
    batch: List[PointStruct] = []
    for index, record in enumerate(records):
        vector, payload = _split_embedding(record, embedding_key)
        if not vector:
            continue
        point_id = _point_id(id_prefix, record, index)
        batch.append(PointStruct(id=point_id, vector=vector, payload=payload))
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection, points=batch)
            total += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)

    return total


def ingest_to_qdrant(
    chunks_json: Path,
    images_json: Path,
    url: str,
    text_collection: str,
    image_collection: str,
    batch_size: int,
    recreate: bool,
) -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required. Install it with: pip install qdrant-client"
        ) from exc

    chunks = _read_json(chunks_json)
    images = _read_json(images_json)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a JSON list")
    if not isinstance(images, list):
        raise ValueError("image_embeddings.json must contain a JSON list")

    text_vector = _first_vector(chunks, "embedding")
    image_vector = _first_vector(images, "embedding")
    if not text_vector:
        raise ValueError("No text embeddings found in chunks.json")
    if not image_vector:
        raise ValueError("No image embeddings found in image_embeddings.json")

    client = QdrantClient(url=url)

    _ensure_collection(client, text_collection, vector_size=len(text_vector), recreate=recreate)
    _ensure_collection(client, image_collection, vector_size=len(image_vector), recreate=recreate)

    text_count = _upsert_records(
        client,
        collection=text_collection,
        records=chunks,
        embedding_key="embedding",
        id_prefix="text",
        batch_size=batch_size,
    )
    image_count = _upsert_records(
        client,
        collection=image_collection,
        records=images,
        embedding_key="embedding",
        id_prefix="image",
        batch_size=batch_size,
    )

    print(f"Upserted {text_count} text points into '{text_collection}'")
    print(f"Upserted {image_count} image points into '{image_collection}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest embeddings into Qdrant.")
    parser.add_argument(
        "--chunks-json",
        default="extracted/chunks.json",
        help="Path to chunks.json",
    )
    parser.add_argument(
        "--images-json",
        default="extracted/image_embeddings.json",
        help="Path to image_embeddings.json",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant URL",
    )
    parser.add_argument(
        "--text-collection",
        default="vlm_text_chunks",
        help="Collection name for text chunks",
    )
    parser.add_argument(
        "--image-collection",
        default="vlm_images",
        help="Collection name for images",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collections if vector sizes mismatch",
    )
    args = parser.parse_args()

    ingest_to_qdrant(
        chunks_json=Path(args.chunks_json),
        images_json=Path(args.images_json),
        url=args.url,
        text_collection=args.text_collection,
        image_collection=args.image_collection,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
