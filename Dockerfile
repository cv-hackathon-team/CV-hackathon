# DepthGuard360 runtime image.
# Files are copied explicitly so .env, outputs/ and checkpoints/ stay out of the image.
#
# dataset_root (../DEPTHGUARD360_DATASETS) resolves to the mount point below:
#   docker run -p 8501:8501 \
#     -v /path/to/DEPTHGUARD360_DATASETS:/workspace/DEPTHGUARD360_DATASETS:ro \
#     -v /path/to/checkpoints:/workspace/depthguard360/checkpoints:ro \
#     depthguard360
FROM python:3.14.6-slim

WORKDIR /workspace/depthguard360
ENV PYTHONPATH=/workspace/depthguard360

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY main.py .

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
