# Vision-Language Model (VLM)

This workspace contains utilities to extract PDF assets and build text/image embeddings.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Extract text, images, and tables

```bash
python pdf_extract.py /path/to/your.pdf --output-dir extracted
```

Outputs:
- extracted/texts.json
- extracted/images.json
- extracted/tables.json
- extracted/manifest.json
- extracted/images/*.png
- extracted/tables/*.png

## Build chunks and embeddings

```bash
python chunk_and_embed.py \
  --texts-json extracted/texts.json \
  --images-json extracted/images.json \
  --output-dir extracted
```

Outputs:
- extracted/chunks.json
- extracted/image_embeddings.json
