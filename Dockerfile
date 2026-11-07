# DepthGuard360 runtime image.
# Files are copied explicitly so .env, outputs/ and checkpoints/ stay out of the image.
FROM python:3.14.6-slim

WORKDIR /workspace/depthguard360
ENV PYTHONPATH=/workspace/depthguard360

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY ui ./ui
COPY outputs/results.csv ./outputs/results.csv
COPY main.py .

EXPOSE 8080
CMD ["uvicorn", "ui.server:app", "--host", "0.0.0.0", "--port", "8080"]
