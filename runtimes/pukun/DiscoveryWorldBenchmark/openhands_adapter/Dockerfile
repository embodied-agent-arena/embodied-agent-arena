ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SDL_VIDEODRIVER=dummy
ENV PYGAME_HIDE_SUPPORT_PROMPT=1

WORKDIR /robench/DiscoveryWorldBenchmark

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY discoveryworld ./discoveryworld
COPY src ./src
COPY scripts ./scripts
COPY openhands_adapter ./openhands_adapter

RUN mkdir -p /robench/scripts \
    && cp ./scripts/openhands_bridge.py /robench/scripts/openhands_bridge.py

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ./discoveryworld

VOLUME ["/robench/DiscoveryWorldBenchmark/outputs"]

CMD ["python", "openhands_adapter/run_instance.py", "--instance-id", "discoveryworld_debug_0"]
