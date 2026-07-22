FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY migrations/ ./migrations/
# mm_committee.py resolves skill files at <repo-root>/docs/agent-committee/skills
# (parents[2] of src/strategy/). Without this the committee raises
# FileNotFoundError on its first call in the container.
COPY docs/agent-committee/ ./docs/agent-committee/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "-m", "src.main"]
