FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifacts
COPY api.py .
COPY model_registry/ ./model_registry/

# Expose port
EXPOSE 8080

# Start API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
