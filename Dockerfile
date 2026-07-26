FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_service/ ai_service/
COPY models/ models/

ENV ESCALATION_THRESHOLD=0.20
ENV MAX_INPUT_LENGTH=5000
ENV INTENT_MODEL_VERSION=intent-v1
ENV ESCALATION_MODEL_VERSION=escalation-v1

EXPOSE 8000

CMD ["uvicorn", "ai_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
