# Build from the depthguard360 repo root:  docker build -f ui/Dockerfile -t depthguard-ui .
# Hugging Face Spaces: copy this file to the repo root as "Dockerfile".
FROM python:3.14-slim
# Spaces run as user 1000; give it a home so caches can be written.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY ui/requirements-ui.txt ui/requirements-ui.txt
RUN pip install --no-cache-dir -r ui/requirements-ui.txt
COPY --chown=user . .
USER user
ENV PORT=7860 HOME=/home/user
EXPOSE 7860
CMD uvicorn ui.server:app --host 0.0.0.0 --port ${PORT}
