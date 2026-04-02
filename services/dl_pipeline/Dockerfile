FROM python:3.10-slim

WORKDIR /app

# Copy code
COPY services/dl_pipeline/src ./services/dl_pipeline/src
COPY services/dl_pipeline/requirement_inference.txt .
COPY data/processed/graph_data.pt ./data/processed/

# Install deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.2.1+cpu.html && \
    pip install torch-geometric && \
    pip install --no-cache-dir -r requirement_inference.txt

ENV PYTHONPATH=/app/services/dl_pipeline/src

CMD ["python", "services/dl_pipeline/src/inference/inference.py"]