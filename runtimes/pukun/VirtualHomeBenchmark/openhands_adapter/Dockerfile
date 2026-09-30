ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUALHOME_OFFICIAL_LIMIT=30

WORKDIR /robench/VirtualHomeBenchmark

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY virtualhome ./virtualhome
COPY src ./src
COPY scripts ./scripts
COPY openhands_adapter ./openhands_adapter

RUN mkdir -p /robench/scripts \
    && cp ./scripts/openhands_bridge.py /robench/scripts/openhands_bridge.py

RUN python -m pip install --upgrade pip \
    && python -m pip install "numpy>=1.26,<3" termcolor==1.1.0 tqdm==4.66.3

VOLUME ["/robench/VirtualHomeBenchmark/outputs"]

CMD ["python", "openhands_adapter/run_instance.py", "--instance-id", "virtualhome_symbolic_debug_0"]
