FROM python:3.9-slim
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Install Python and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    build-essential \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Create a directory for the app
WORKDIR /opt/ml

# Copy your model and inference code
COPY code /opt/ml/code

# Install Python dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -v -r /opt/ml/code/requirements.txt

# Set the environment variables for SageMaker
ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Define the entry point for the container
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]