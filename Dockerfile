FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install the official CPU-only PyTorch wheel first. The matching requirement
# is then already satisfied when the rest of the application stack is installed.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/storage && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT', '8501'), timeout=3)"

CMD ["sh", "-c", "streamlit run app/Main.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
