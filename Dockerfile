FROM python:3.12-slim

WORKDIR /app

# Copy deps first so Docker can cache this layer — pip install only re-runs
# when requirements.txt changes, not on every source file edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command is overridden per service in docker-compose.yml.
# --host 0.0.0.0 is required inside Docker; 127.0.0.1 would only be reachable
# within the container itself, not from the host or other containers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
