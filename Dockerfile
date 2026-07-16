# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (helps with Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY src/ ./src/
COPY data/ ./data/

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]