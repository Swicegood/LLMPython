# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Pre-set the timezone to avoid prompts
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive 

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
RUN pip3 install --no-cache-dir \
    flask \
    transformers \
    pillow \
    requests \
    bitsandbytes \
    accelerate \
    wheel \
    scipy \
    sentencepiece \
    peft

# Make port 1234 available to the world outside this container
EXPOSE 1234

# Run app.py when the container launches
CMD ["python3", "app.py"]