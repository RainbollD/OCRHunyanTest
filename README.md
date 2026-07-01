# Hunyuan OCR — сервис распознавания текста

Распознавание текста моделью [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
(end-to-end OCR VLM, 1B параметров) через **vLLM** с OpenAI-совместимым HTTP API.
Контейнер поднимает сервер, обращение — по `curl` или тонким клиентом `main.py`.

## Архитектура

- **Сервер** (контейнер): `vllm serve tencent/HunyuanOCR` — модель постоянно в VRAM,
  отдаёт `POST /v1/chat/completions`. Разворачивается официальным образом `vllm/vllm-openai`.
- **Клиент**: любой HTTP — `curl`, `main.py` или ваш бэкенд. Картинка передаётся
  как base64 `data:`-URL.
- **PDF**: HunyuanOCR — vision-модель (читает картинки, не PDF). `main.py` сам
  растеризует PDF постранично (через `pypdfium2`) и шлёт каждую страницу как изображение.

## Требования

- NVIDIA GPU. **Целевая карта — 16 ГБ.** ⚠️ Официальный минимум Tencent для vLLM —
  **20 ГБ**, то есть 16 ГБ ниже спецификации: для одиночных запросов работает при
  поджатых настройках, но требует проверки на вашей карте (см. [Тюнинг под 16 ГБ](#тюнинг-под-16-гб)).
- Драйвер NVIDIA под CUDA 12.9+ (образ `v0.24.0` собран под CUDA 12.9). Проверка: `nvidia-smi`.
- Docker + Docker Compose v2 + `nvidia-container-toolkit`.
- vLLM ≥ 0.12.0 (задаётся тегом образа в `.env`, по умолчанию `v0.24.0`).

## Быстрый старт

```bash
cp .env.example .env          # при желании поправьте настройки
docker compose up -d          # первый запуск скачает ~2-6 ГБ модели
docker compose logs -f ocr    # дождитесь "Application startup complete"
docker compose ps             # STATUS = healthy → готово
```

Проверка:

```bash
curl http://localhost:8000/v1/models
./curl_example.sh path/to/image.png
```

## Использование

### curl

```bash
IMG=scan.png
B64=$(base64 -w0 "$IMG")
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hunyuan-ocr",
    "temperature": 0,
    "max_tokens": 4096,
    "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$B64"'"}},
      {"type":"text","text":"提取文档图片中正文的所有信息用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"}
    ]}]
  }'
```

Готовый скрипт: `./curl_example.sh image.png`.

### Клиент main.py

```bash
pip install -r requirements.txt

python main.py scan.png                                  # чистый текст/markdown (task=parsing)
python main.py scan.png --task spotting                  # весь текст + координаты
python main.py receipt.jpg --task extract --fields "продавец,дата,сумма"
python main.py photo.jpg --max-side 2048                 # даунскейл для скорости
python main.py scan.png --prompt "..."                   # свой промпт
```

Клиент печатает результат в stdout, а время ответа — в stderr (`[page: 2.13s]`),
удобно мерить попадание в 2-3 с.

### PDF

```bash
python main.py doc.pdf                        # все страницы
python main.py doc.pdf --pages "1-3,5,8-"     # выбранные страницы (нумерация с 1)
python main.py doc.pdf --dpi 200              # разрешение растеризации (150-300; выше = точнее, но медленнее)
python main.py doc.pdf --workers 4            # слать N страниц параллельно (<= MAX_NUM_SEQS сервера)
python main.py doc.pdf --out result.md        # весь документ одним md-файлом
```

`--workers` кратно ускоряет многостраничные PDF, но не ставь больше, чем `MAX_NUM_SEQS`
у сервера (по умолчанию 4) — иначе запросы встанут в очередь.

### Формат итога

С `--task parsing` (по умолчанию) каждая страница возвращается **структурированным
markdown**: заголовки, списки, формулы (LaTeX), в порядке чтения. С `--out file.md`
всё собирается в **один md-файл**.

- **Таблицы** по умолчанию — в HTML (точнее для объединённых ячеек, валидно внутри md).
  Нужны md-«трубы» `| a | b |` — добавь `--md-tables`.
- **Разделитель страниц** (`--page-sep`): `rule` — `---` (по умолчанию), `heading` —
  `## page N`, `comment` — невидимый `<!-- page N -->`, `none` — сплошной текст.
- Каждая страница распознаётся **независимо**: таблица или абзац, разорванные между
  страницами, автоматически не сшиваются (ограничение постраничного OCR).

### Задачи (промпты Tencent)

| task | что делает |
| ---- | ---------- |
| `parsing` (по умолчанию) | текст документа → markdown, таблицы → HTML, формулы → LaTeX, в порядке чтения (игнорирует колонтитулы) |
| `spotting` | детекция + распознавание всего текста с координатами; самый полный вывод |
| `extract` | извлечение полей (`--fields`) в JSON |

## Тюнинг под 16 ГБ

Настройки в `.env`, применяются пересозданием контейнера (`docker compose up -d`):

| Переменная | Смысл | Если OOM |
| ---------- | ----- | -------- |
| `GPU_MEMORY_UTILIZATION` | доля VRAM под vLLM | оставить ~0.90 |
| `MAX_MODEL_LEN` | макс. контекст (vision + промпт + вывод) | понизить (напр. `8192`) |
| `MAX_NUM_SEQS` | одновременных запросов | понизить (напр. `2`) |

Клиентские рычаги: `--max-side 2048` (меньше vision-токенов → быстрее prefill и меньше памяти),
`--max-tokens 4096` (не держать потолок 16384).

Если всё равно не влезает — добавьте в `command` (docker-compose.yml) флаг `--enforce-eager`
(отключит CUDA-графы: экономит память ценой ~10-20% скорости).

## Скорость (цель 2-3 с/страница)

Достижимо для типичной страницы; зависит от карты и объёма текста (декодирование
авторегрессивное — плотная страница = дольше). Основные рычаги: vLLM (уже используется),
лимит `--max-tokens`, `--max-side` для контроля разрешения. Абсолютный максимум скорости —
TensorRT-LLM, но это отдельная, более сложная сборка.

## Troubleshooting

- **OOM при старте/запросе** — см. [Тюнинг под 16 ГБ](#тюнинг-под-16-гб): начните с
  `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=2`, при необходимости `--enforce-eager`.
- **`requires trust_remote_code`** — добавьте в `command` флаг `--trust-remote-code`.
- **CUDA / driver mismatch** — образ собран под CUDA 12.9; обновите драйвер
  (`nvidia-smi` → CUDA ≥ 12.9) или возьмите тег постарше в `VLLM_TAG` (не ниже `v0.12.0`).
- **Долгий первый старт** — качается модель (~2-6 ГБ); healthcheck ждёт до 15 мин (`start_period`).
- **`Model architecture not supported`** — `VLLM_TAG` должен быть ≥ `v0.12.0`.

## Заметка про dev-машину

RTX 3050 Laptop (4 ГБ) полноценно модель не запустит — не хватит VRAM.
Разворачивайте на целевой 16 ГБ карте.
