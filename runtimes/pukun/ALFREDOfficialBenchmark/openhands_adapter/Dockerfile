ARG PYTHON_BASE_IMAGE=python:3.8
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ALFWORLD_DATA=/robench/ALFWorldBenchmark/data
ENV PYTHONPATH=/robench/ALFREDOfficialBenchmark/src:/robench/ALFWorldBenchmark/alfworld:/robench/scripts

WORKDIR /robench/ALFREDOfficialBenchmark

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl wget xvfb xauth libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY src ./src
COPY scripts ./scripts
COPY openhands_adapter ./openhands_adapter

RUN mkdir -p /robench/scripts \
    && cp ./scripts/openhands_bridge.py /robench/scripts/openhands_bridge.py

RUN python -m pip install --upgrade "pip<25" \
    && python -m pip install ai2thor==2.1.0 opencv-python-headless pandas tqdm werkzeug==2.0.3 pyyaml networkx

VOLUME ["/robench/ALFREDOfficialBenchmark/outputs", "/robench/ALFWorldBenchmark"]

CMD ["python", "openhands_adapter/run_instance.py", "--instance-id", "alfred_official_debug_0"]
