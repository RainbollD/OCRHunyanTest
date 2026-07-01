#!/usr/bin/env bash
# Minimal curl example: OCR an image through the HunyuanOCR vLLM server.
# Usage: ./curl_example.sh IMAGE [BASE_URL]
set -euo pipefail

IMG="${1:?usage: ./curl_example.sh IMAGE [BASE_URL]}"
BASE="${2:-http://localhost:8000}"

case "${IMG,,}" in
  *.png)        MIME=image/png ;;
  *.jpg|*.jpeg) MIME=image/jpeg ;;
  *.webp)       MIME=image/webp ;;
  *)            MIME=image/png ;;
esac

B64=$(base64 -w0 "$IMG")
# "parsing" task: document body -> markdown, tables -> HTML, formulas -> LaTeX.
PROMPT='提取文档图片中正文的所有信息用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。'

curl -s "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<JSON | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
{
  "model": "hunyuan-ocr",
  "temperature": 0,
  "max_tokens": 4096,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:${MIME};base64,${B64}"}},
    {"type": "text", "text": "${PROMPT}"}
  ]}]
}
JSON
