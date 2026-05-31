# Hunyuan OCR Test

Тестовый запуск модели [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) через Transformers.

## Требования

- Linux
- NVIDIA GPU с CUDA
- Python 3.12+ (рекомендуется; в Docker — 3.13)
- ~6 ГБ на диске для модели

## Локальный запуск

```bash
cd OCRHunyanTest

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`torch` и `torchvision` установятся автоматически как зависимости (`accelerate` → `torch`, в requirements есть `torchvision`). С PyPI приходит сборка с CUDA.

Отдельная установка через `--index-url https://download.pytorch.org/whl/cu124` нужна только если хотите явно указать версию CUDA (как в Docker-образе).

### Запуск

```bash
# по умолчанию — 12.png в папке проекта
python main.py

# свой файл
python main.py /path/to/image.png
python main.py scan.jpg

# справка
python main.py --help
```

При первом запуске модель скачается с Hugging Face. Результат OCR выводится в консоль.

## Запуск через Docker

```bash
docker compose up -d --build
docker exec -it ocr_hunyuan bash
```

Внутри контейнера:

```bash
python main.py
python main.py /app/my_image.png
```

Папка проекта смонтирована в `/app`, кэш модели сохраняется в volume `hunyuan_cache`.
