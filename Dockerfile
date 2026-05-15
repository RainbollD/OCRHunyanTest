FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Чтобы apt не задавал интерактивные вопросы, например про часовой пояс
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# 1. Системные зависимости + PPA для Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    nano \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# 2. Настройка python / pip
RUN python3.13 -m ensurepip --upgrade \
    && ln -sf /usr/bin/python3.13 /usr/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3 \
    && python -m pip install --upgrade pip setuptools wheel

# 3. PyTorch с CUDA 12.4
RUN python -m pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Transformers с конкретного commit
RUN python -m pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4

# 5. Остальные Python-зависимости
RUN python -m pip install --no-cache-dir \
    Pillow \
    huggingface_hub \
    accelerate \
    sentencepiece \
    protobuf

# 6. Копируем проект
COPY . .

# 7. Чтобы контейнер не завершался сразу
CMD ["sleep", "infinity"]