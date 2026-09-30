ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ALFWORLD_DATA=/data/alfworld

WORKDIR /robench/ALFWorldBenchmark

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY alfworld ./alfworld
COPY src ./src
COPY scripts ./scripts
COPY openhands_adapter ./openhands_adapter

RUN mkdir -p /robench/scripts \
    && cp ./scripts/openhands_bridge.py /robench/scripts/openhands_bridge.py

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ./alfworld

VOLUME ["/data/alfworld", "/robench/ALFWorldBenchmark/outputs"]

CMD ["python", "openhands_adapter/run_instance.py", "--instance-id", "alfworld_text_debug_0"]
