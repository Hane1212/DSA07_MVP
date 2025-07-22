# Use slim image with Python
FROM python:3.12-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-multipart

# Copy all project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501
EXPOSE 8000

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
