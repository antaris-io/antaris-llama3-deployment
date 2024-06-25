FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# Install Python and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    build-essential \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Create directories for your code
WORKDIR /opt/ml

# Copy inference code into the container
COPY code /opt/ml/code

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Define the entry point
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]