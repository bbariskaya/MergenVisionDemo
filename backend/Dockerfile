# ---- Builder stage: compile native CUDA extension ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS native-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv cmake ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip

COPY native/mergenvision_gpu /build/native/mergenvision_gpu
WORKDIR /build/native/mergenvision_gpu
RUN /opt/venv/bin/pip install --no-cache-dir pybind11 && \
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=/opt/venv/bin/python3.11 \
        -Dpybind11_DIR=/opt/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 && \
    cmake --build build && \
    mkdir -p /out/mergenvision_gpu && \
    cp build/_mergenvision_gpu*.so /out/mergenvision_gpu/ && \
    cp __init__.py /out/mergenvision_gpu/

# ---- Runtime stage ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    WORKDIR=/app \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv && \
    pip install --no-cache-dir -U pip

COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e ".[dev]"

COPY --from=native-builder /out/mergenvision_gpu /opt/venv/lib/python3.11/site-packages/mergenvision_gpu

COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
