from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _save_pixmap_as_png(pix: fitz.Pixmap, output_path: Path) -> None:
    if pix.width <= 0 or pix.height <= 0:
        return
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Force conversion to RGB if not Gray/RGB or if it has weird properties
        if (pix.colorspace and pix.colorspace.name not in (fitz.csRGB.name, fitz.csGRAY.name)) or pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            
        pix.save(str(output_path))
    except Exception as e:
        try:
            # Fallback using PIL
            from PIL import Image
            mode = "RGBA" if pix.alpha else "RGB"
            if pix.n == 1:
                mode = "L"
            elif pix.n == 2:
                mode = "LA"
            elif pix.n == 4 and not pix.alpha:
                mode = "CMYK"
                
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if mode == "CMYK":
                img = img.convert("RGB")
            img.save(str(output_path))
        except Exception as inner_e:
            print(f"Warning: Could not save image {output_path} - PyMuPDF error: {e}, PIL error: {inner_e}")


def extract_pdf_assets(pdf_path: str, output_dir: str = "extracted") -> Dict[str, Path]:
    """Extract text blocks, images, and tables from a PDF into JSON + PNG files.

    Produces `texts.json`, `images.json`, `tables.json`, and `manifest.json` in the
    `output_dir`. Also writes PNGs for images and table snapshots.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    tables_dir = out_dir / "tables"

    texts: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    text_global_id = 0
    image_global_id = 0
    table_global_id = 0

    pages_manifest: List[Dict[str, Any]] = []

    with fitz.open(pdf_file) as doc:
        total_pages = doc.page_count
        for page_number, page in enumerate(doc, start=1):
            page_prefix = f"p{page_number:03d}"
            page_texts: List[Dict[str, Any]] = []
            page_images: List[Dict[str, Any]] = []
            page_tables: List[Dict[str, Any]] = []

            # Text blocks with bounding boxes.
            per_page_text_idx = 0
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, _, block_type = block
                if block_type != 0:
                    continue
                text = text.strip()
                if not text:
                    continue
                text_global_id += 1
                block_id = f"{page_prefix}_b{per_page_text_idx:02d}"
                per_page_text_idx += 1

                page_texts.append(
                    {
                        "block_id": block_id,
                        "text": text,
                        "page_number": page_number,
                        "bbox": [x0, y0, x1, y1],
                    }
                )

            # Image blocks with bounding boxes.
            block_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            per_page_img_idx = 0
            for block in block_dict.get("blocks", []):
                if block.get("type") != 1:
                    continue

                bbox = block.get("bbox")
                if not bbox:
                    continue

                image_global_id += 1
                img_id = f"{page_prefix}_img_{per_page_img_idx}"
                per_page_img_idx += 1
                image_path = images_dir / f"{img_id}.png"

                xref = block.get("xref")
                try:
                    if xref:
                        pix = fitz.Pixmap(doc, xref)
                    else:
                        pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=200)
                    _save_pixmap_as_png(pix, image_path)
                except Exception:
                    pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=200)
                    _save_pixmap_as_png(pix, image_path)

                page_images.append(
                    {
                        "image_id": img_id,
                        "page_number": page_number,
                        "bbox": list(bbox),
                        "image_path": str(image_path),
                        "xref": xref,
                        "width": block.get("width"),
                        "height": block.get("height"),
                    }
                )

            # Tables (if supported by your PyMuPDF version).
            if hasattr(page, "find_tables"):
                try:
                    table_finder = page.find_tables()
                except Exception:
                    table_finder = None

                if table_finder and getattr(table_finder, "tables", None):
                    per_page_tbl_idx = 0
                    for table in table_finder.tables:
                        table_global_id += 1
                        tbl_id = f"{page_prefix}_tbl_{per_page_tbl_idx}"
                        per_page_tbl_idx += 1
                        bbox = list(table.bbox)
                        table_image_path = tables_dir / f"{tbl_id}.png"
                        pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=200)
                        _save_pixmap_as_png(pix, table_image_path)

                        try:
                            cells = table.extract()
                        except Exception:
                            cells = None

                        page_tables.append(
                            {
                                "table_id": tbl_id,
                                "page_number": page_number,
                                "bbox": bbox,
                                "image_path": str(table_image_path),
                                "cells": cells,
                            }
                        )

            # Compute nearest neighbors on this page (vertical distance based on centers).
            def _y_center(bbox: List[float]) -> float:
                return (bbox[1] + bbox[3]) / 2.0

            # For each image, find nearest two text blocks vertically.
            for img in page_images:
                img_center_y = _y_center(img["bbox"])
                distances = []
                for t in page_texts:
                    dist = abs(img_center_y - _y_center(t["bbox"]))
                    distances.append((dist, t["block_id"]))
                distances.sort(key=lambda x: x[0])
                nearest = [bid for _, bid in distances[:2]]
                img["nearest_blocks"] = nearest

            # For each text block, find nearest two images vertically.
            for t in page_texts:
                t_center_y = _y_center(t["bbox"])
                distances = []
                for img in page_images:
                    dist = abs(t_center_y - _y_center(img["bbox"]))
                    distances.append((dist, img["image_id"]))
                distances.sort(key=lambda x: x[0])
                nearest_imgs = [iid for _, iid in distances[:2]]
                t["nearby_images"] = nearest_imgs

            # Extend global lists and build page manifest entry.
            texts.extend(page_texts)
            images.extend(page_images)
            tables.extend(page_tables)

            pages_manifest.append(
                {
                    "page_number": page_number,
                    "text_block_ids": [t["block_id"] for t in page_texts],
                    "image_ids": [i["image_id"] for i in page_images],
                    "table_ids": [tb["table_id"] for tb in page_tables],
                }
            )

    _write_json(out_dir / "texts.json", texts)
    _write_json(out_dir / "images.json", images)
    _write_json(out_dir / "tables.json", tables)

    manifest = {
        "source_file": pdf_file.name,
        "total_pages": int(total_pages),
        "total_text_blocks": len(texts),
        "total_images": len(images),
        "total_tables": len(tables),
        "pages": pages_manifest,
    }

    _write_json(out_dir / "manifest.json", manifest)

    return {
        "texts_json": out_dir / "texts.json",
        "images_json": out_dir / "images.json",
        "tables_json": out_dir / "tables.json",
        "manifest_json": out_dir / "manifest.json",
        "images_dir": images_dir,
        "tables_dir": tables_dir,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract PDF text, images, and tables.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output-dir", default="extracted", help="Directory for extracted assets"
    )
    args = parser.parse_args()

    extract_pdf_assets(args.pdf_path, args.output_dir)
