ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SCIENCEWORLD_SIMPLIFICATION=easy

WORKDIR /robench/ScienceWorldBenchmark

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl default-jre \
    && rm -rf /var/lib/apt/lists/*

COPY scienceworld ./scienceworld
COPY src ./src
COPY scripts ./scripts
COPY openhands_adapter ./openhands_adapter

RUN mkdir -p /robench/scripts \
    && cp ./scripts/openhands_bridge.py /robench/scripts/openhands_bridge.py

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ./scienceworld

VOLUME ["/robench/ScienceWorldBenchmark/outputs"]

CMD ["python", "openhands_adapter/run_instance.py", "--instance-id", "scienceworld_text_debug_0"]
