# API service — PyTerrier requires Java; NeuralReranker requires torch
FROM python:3.13-slim

# Java (Terrier/JPype) + C build toolchain (compiling trec_eval for pytrec_eval_terrier)
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless \
        curl \
        build-essential \
        gcc \
        make \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# pytrec_eval_terrier's setup.py downloads trec_eval source from GitHub at build time.
# Install it separately with retries to tolerate transient network drops.
RUN pip install --no-cache-dir --timeout 120 pytrec_eval_terrier || \
    (sleep 20 && pip install --no-cache-dir --timeout 120 pytrec_eval_terrier) || \
    (sleep 40 && pip install --no-cache-dir --timeout 120 pytrec_eval_terrier)

# Install remaining deps (pip skips pytrec_eval_terrier since it's already present)
COPY requirements.api.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.api.txt

COPY src/ ./src/
COPY api/ ./api/
COPY main.py .

# Index and PyTerrier JAR cache live in mounted volumes
ENV INDEX_PATH=/data/index
ENV PYTERRIER_NO_CHECK=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
