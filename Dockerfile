FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir '.[dev,phase1]'
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
CMD ["python", "scripts/check.py"]
