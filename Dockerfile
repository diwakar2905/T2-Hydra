FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build deps required for some binary wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements into the image
COPY requirements.txt ./requirements.txt
# Also copy service-level requirements so -r references resolve
COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Final image: copy installed packages from builder
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Default command (Render overrides with startCommand)
CMD ["bash", "-lc", "cd backend && gunicorn app:app --bind 0.0.0.0:$PORT"]
