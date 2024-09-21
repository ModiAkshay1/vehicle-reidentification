# Use the CUDA base image with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

ENV TZ=India \
    DEBIAN_FRONTEND=noninteractive
# Install system dependencies and Python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
    wget \
    bzip2 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python packages using pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Set the entrypoint or command to run your application
# For example, if you have a script named main.py, you can set it as:
# CMD ["python3", "main.py"]