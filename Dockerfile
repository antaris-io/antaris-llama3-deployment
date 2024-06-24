FROM python:3.9-slim

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