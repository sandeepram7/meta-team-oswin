FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any (e.g., for scikit-learn or fireducks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH to root so 'from models import ...' works inside server/
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
