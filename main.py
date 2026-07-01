#!/usr/bin/env python3
"""HunyuanOCR client.

Thin client that sends an image (or every page of a PDF) to the vLLM
OpenAI-compatible server (see docker-compose.yml) and prints the recognized text.
The model runs in the container; this script just talks to it over HTTP.

HunyuanOCR is a vision model — it reads images, not PDFs — so PDFs are
rasterized page by page (via pypdfium2) and each page is sent as an image.
"""
import argparse
import base64
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Task prompts recommended by Tencent for HunyuanOCR.
# https://github.com/Tencent-Hunyuan/HunyuanOCR
# Text spotting: detect and recognize every piece of text, output with coordinates.
SPOTTING_PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"


def parsing_prompt(md_tables=False):
    """Document parsing -> structured markdown (headings, lists, tables, formulas) in reading order."""
    table_fmt = "markdown" if md_tables else "html"
    return (
        f"提取文档图片中正文的所有信息用markdown格式表示，表格用{table_fmt}格式表达，"
        f"文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    )


def build_extract_prompt(fields):
    """Build the field-extraction prompt from a comma-separated list of field names."""
    keys = ",".join(f"'{f.strip()}'" for f in fields.split(",") if f.strip())
    return f"提取图片中的: [{keys}] 的字段内容，并按照JSON格式返回。"


def parse_pages(spec, total):
    """Turn a '1-3,5,8-' page spec (1-based, inclusive) into a list of 0-based indices."""
    idx = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a else 1
            end = int(b) if b else total
        else:
            start = end = int(part)
        idx.extend(range(start - 1, end))
    return [i for i in idx if 0 <= i < total]


def image_to_data_url(img, max_side=0):
    """Encode a PIL image as a PNG data: URL, optionally downscaling so max(w, h) <= max_side."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max_side and max_side > 0:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def load_pages(path, dpi, pages_spec, max_side):
    """Return [(label, data_url), ...]: one entry per image, one per rasterized PDF page."""
    if path.lower().endswith(".pdf"):
        import pypdfium2 as pdfium  # only needed for PDF input

        pdf = pdfium.PdfDocument(path)
        total = len(pdf)
        indices = parse_pages(pages_spec, total) if pages_spec else range(total)
        out = []
        for i in indices:
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72.0)  # pdfium renders at 72 DPI * scale
            out.append((f"page {i + 1}/{total}", image_to_data_url(bitmap.to_pil(), max_side)))
            bitmap.close()
            page.close()
        pdf.close()
        return out

    # Single image file: keep the original bytes/format unless we need to downscale.
    with open(path, "rb") as f:
        raw = f.read()
    if max_side and max_side > 0:
        from PIL import Image

        return [(os.path.basename(path), image_to_data_url(Image.open(BytesIO(raw)), max_side))]
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return [(os.path.basename(path), f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}")]


def ocr_one(session, endpoint, model, prompt, data_url, max_tokens, timeout):
    """Send one image to the API and return the recognized text."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    resp = session.post(endpoint, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Server returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()["choices"][0]["message"]["content"]


def parse_args():
    p = argparse.ArgumentParser(description="HunyuanOCR client (image or PDF -> text via the vLLM server)")
    p.add_argument("input", help="Path to an image or a .pdf file")
    p.add_argument(
        "--task",
        choices=["parsing", "spotting", "extract"],
        default="parsing",
        help="parsing (default): clean markdown text; spotting: all text + coordinates; "
        "extract: fields as JSON (needs --fields)",
    )
    p.add_argument("--fields", help="Comma-separated fields for --task extract, e.g. 'name,date,total'")
    p.add_argument("--prompt", help="Custom prompt (overrides --task)")
    p.add_argument("--pages", help="PDF page range (1-based), e.g. '1-3,5,8-'. Default: all pages")
    p.add_argument("--dpi", type=int, default=200, help="PDF rasterization DPI (default: %(default)s; 150-300 useful)")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests for multi-page PDFs (default: 1). Keep <= server MAX_NUM_SEQS.",
    )
    p.add_argument(
        "--url",
        default=os.environ.get("OCR_URL", "http://localhost:8000/v1"),
        help="Base URL of the OpenAI-compatible API (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OCR_MODEL", "hunyuan-ocr"),
        help="Served model name (default: %(default)s)",
    )
    p.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens (default: %(default)s)")
    p.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Downscale so the longest side <= N px (0 = off). Lower = faster; needs Pillow.",
    )
    p.add_argument("--timeout", type=int, default=180, help="HTTP timeout, seconds (default: %(default)s)")
    p.add_argument("--out", help="Write recognized text to this file instead of stdout")
    p.add_argument(
        "--page-sep",
        choices=["rule", "heading", "comment", "none"],
        default="rule",
        help="How to separate PDF pages in the merged output: rule (---, default), "
        "heading (## page N), comment (<!-- page N -->), none",
    )
    p.add_argument(
        "--md-tables",
        action="store_true",
        help="Ask for markdown pipe-tables instead of HTML tables (parsing task). "
        "HTML (default) is higher fidelity for complex tables.",
    )
    args = p.parse_args()
    if not os.path.isfile(args.input):
        p.error(f"File not found: {args.input}")
    if args.task == "extract" and not (args.fields or args.prompt):
        p.error("--task extract requires --fields (or pass --prompt)")
    return args


def main():
    args = parse_args()
    import requests  # imported after arg parsing so --help works without the dependency

    if args.prompt:
        prompt = args.prompt
    elif args.task == "extract":
        prompt = build_extract_prompt(args.fields)
    elif args.task == "spotting":
        prompt = SPOTTING_PROMPT
    else:  # parsing
        prompt = parsing_prompt(args.md_tables)

    try:
        pages = load_pages(args.input, args.dpi, args.pages, args.max_side)
    except ImportError:
        sys.exit("PDF support needs pypdfium2:  pip install pypdfium2")
    if not pages:
        sys.exit("No pages selected (check --pages).")

    endpoint = args.url.rstrip("/") + "/chat/completions"
    session = requests.Session()

    def work(item):
        label, data_url = item
        t0 = time.time()
        text = ocr_one(session, endpoint, args.model, prompt, data_url, args.max_tokens, args.timeout)
        return label, text, time.time() - t0

    t_start = time.time()
    try:
        if args.workers > 1 and len(pages) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(work, pages))
        else:
            results = [work(p) for p in pages]
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Cannot reach the OCR server at {endpoint}.\n"
            "Is the container up and healthy?  docker compose up -d  (then wait for the model to load)"
        )
    except RuntimeError as e:
        sys.exit(str(e))

    # Assemble one structured markdown document from the per-page results.
    multi = len(results) > 1
    parts = []
    for label, text, dt in results:
        if multi and args.page_sep == "heading":
            parts.append(f"## {label}\n\n{text}")
        elif multi and args.page_sep == "comment":
            parts.append(f"<!-- {label} -->\n\n{text}")
        else:
            parts.append(text)
        print(f"[{label}: {dt:.2f}s]", file=sys.stderr)
    joiner = "\n\n---\n\n" if (multi and args.page_sep == "rule") else "\n\n"
    body = joiner.join(parts)

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        print(body, file=out)
    finally:
        if args.out:
            out.close()
    if multi:
        print(f"[total {len(results)} pages: {time.time() - t_start:.2f}s]", file=sys.stderr)


if __name__ == "__main__":
    main()
