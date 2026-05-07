FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal system deps for wheels and runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy app source
COPY . .

# Default command (Render overrides with startCommand)
CMD ["bash", "-lc", "gunicorn backend.app:app --bind 0.0.0.0:$PORT"]
