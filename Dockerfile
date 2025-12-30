# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen

# Copy application code
COPY *.py ./
COPY *.pkl ./
COPY *.pth ./

# Expose the port the app runs on
EXPOSE 8080

# Set Python path to include the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run the FastAPI server
CMD ["python", "predict.py"]

