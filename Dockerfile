FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libsndfile1 sox ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# numpy + torch first — sox needs numpy at install time
RUN pip install --no-cache-dir numpy torch torchaudio

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
