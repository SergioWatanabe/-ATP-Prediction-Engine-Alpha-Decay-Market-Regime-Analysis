# Use Python 3.12 base image (safer for package compatibility)
FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libsm6 libxext6 cmake rsync libgl1 \
    curl build-essential python3-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    gradio[oauth,mcp]==6.11.0 \
    "uvicorn>=0.14.0" "websockets>=10.4" \
    spaces

CMD ["python", "app.py"]