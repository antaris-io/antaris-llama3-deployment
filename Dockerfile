FROM python:3.9-slim

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -v -r requirements.txt

# Create a directory for the app
WORKDIR /opt/ml

# Copy your model and inference code
COPY Meta-Llama-3-8B-hf /opt/ml/Meta-Llama-3-8B-hf
COPY Meta-Llama-3-8B-hf-finetuned /opt/ml/Meta-Llama-3-8B-hf
COPY code /opt/ml/code

# Set the environment variables for SageMaker
ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Define the entry point for the container
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]