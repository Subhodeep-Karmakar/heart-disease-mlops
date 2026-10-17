# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables to optimize Python for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best_model.pkl

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, model, and application
COPY src/ /app/src/
COPY models/ /app/models/
COPY app/ /app/app/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI serving application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]