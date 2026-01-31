FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Long timeout (1h) for large downloads (torch ~915 MB); retries on flaky networks
RUN pip3 install --no-cache-dir --timeout 3600 --retries 5 -r requirements.txt

COPY app/ ./app/

EXPOSE 8030

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8030", "--timeout-keep-alive", "3600"]
