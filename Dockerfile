# 1. Use an official Python base image
FROM python:3.10-slim

# 2. Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to leverage Docker's cache
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Create the mlruns directory inside the container
RUN mkdir -p mlruns

# 9. Default command: Run the tests to ensure the container is healthy
CMD ["python", "-m", "pytest", "tests/test_train.py"]
