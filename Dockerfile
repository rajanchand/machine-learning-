FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (compiler, git, etc. if needed, but scikit-learn wheel doesn't need much)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

# Expose FastAPI port and Ingestion socket port
EXPOSE 8000
EXPOSE 9999

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
