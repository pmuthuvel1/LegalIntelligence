FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    LEGAL_DATA_DIR=/app/data \
    LEGAL_LOGS_DIR=/app/logs

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r legal && useradd -r -g legal legal

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/logs /app/logs/checkpoints \
    && chmod +x entrypoint.sh \
    && chown -R legal:legal /app

USER legal

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8001/v1/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
