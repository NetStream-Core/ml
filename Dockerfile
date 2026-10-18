FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./

RUN uv sync --locked --no-dev

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:${PATH}"
ENV KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
ENV KAFKA_EVENTS_TOPIC="netstream.events.v1"
ENV KAFKA_DETECTIONS_TOPIC="netstream.detections.v1"
ENV KAFKA_CONSUMER_GROUP="netstream-detector"

ENTRYPOINT ["netstream-ml", "detect", "run"]
