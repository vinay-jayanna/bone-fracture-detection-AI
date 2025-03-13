# Use an official MLServer base image
FROM python:3.11.2

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir ultralytics torch torchvision torchaudio mlserver mlflow opencv-python-headless

# Copy necessary files
COPY custom_runtime.py /app/custom_runtime.py
COPY model-settings.json /app/model-settings.json
COPY runs/detect/train/weights/best.pt /app/model/best.pt



# Expose the MLServer default port
EXPOSE 8080

# Start MLServer
CMD ["mlserver", "start", "/app"]
