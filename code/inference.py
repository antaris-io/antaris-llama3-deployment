"""
Template taken from:
https://towardsdatascience.com/deploy-a-custom-ml-model-as-a-sagemaker-endpoint-6d2540226428
"""
# local imports
from jsonformer import Jsonformer
from processing import Preprocessor

# external imports
import json
import os
from transformers import  AutoTokenizer
from vllm import LLM
import boto3
from cloudpathlib import CloudPath
import logging


# Define the S3 bucket and model path
S3_BUCKET = "s3://253333439226-app-registry/llama3"
S3_BASE_MODEL_PATH = "Meta-Llama-3-8B-hf"
S3_MODEL_PATH = "Meta-Llama-3-8B-hf-finetuned"

# define the paths
BASEPATH = os.path.dirname(__file__)
BASE_MODEL = os.path.join(os.path.dirname(BASEPATH), "Meta-Llama-3-8B-hf")
MODEL_PATH = os.path.join(os.path.dirname(BASEPATH), "Meta-Llama-3-8B-hf-finetuned")
JSON_SCHEME_PATH = os.path.join(BASEPATH, "json_scheme.json")
PROMP_PATH = os.path.join(BASEPATH, "prompt.txt")
MODEL_SIZE = 4*2048
USE_RELAY = True



# ============================================================
# SageMaker Inference Functions
# ============================================================

def download_model_from_s3():
    cp = CloudPath(f"{S3_BUCKET}/{S3_BASE_MODEL_PATH}")
    cp.download_to(BASE_MODEL)

    cp = CloudPath(f"{S3_BUCKET}/{S3_MODEL_PATH}")
    cp.download_to(MODEL_PATH)


def model_fn(model_dir):
    """
    This function is the first to get executed upon a prediction request,
    it loads the model from the disk and returns the model object which will be used later for inference.
    """

    # Download model from S3
    logging.debug("Downloading model from S3...")
    download_model_from_s3()

    # Load the model
    logging.debug("Loading model to GPU...")
    model = LLM(BASE_MODEL, enable_lora=True, max_lora_rank=64)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    return model, tokenizer


def input_fn(text, request_content_type):
    """
    The request_body is passed in by SageMaker and the content type is passed in 
    via an HTTP header by the client (or caller).
    """
    logging.debug("Preprocess input...")

    if request_content_type == "text/csv":
        return text

    else:
        # If the request_content_type is not as expected, raise an exception
        raise ValueError(f"Content type {request_content_type} is not supported")


def predict_fn(prompt, model_tokenizer):
    """
    This function takes in the input data and the model returned by the model_fn
    It gets executed after the model_fn and its output is returned as the API response.
    """
    logging.debug("Predicting result...")

    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]

    # Load raw prompt
    with open(PROMP_PATH, "r") as f:
        prompt_raw = f.read()

    # Load json scheme
    with open(JSON_SCHEME_PATH, "r") as f:
        json_scheme = json.load(f)

    # Preprocess the input
    tokens_offset = len(tokenizer.encode(f"""{prompt_raw.format(text="")} {str(json_scheme)}"""))
    processor = Preprocessor(tokenizer, MODEL_SIZE-2*tokens_offset)
    prompt = processor(prompt)

    # Initialize the Jsonformer
    jsonformer = Jsonformer(model, tokenizer, json_scheme, prompt, temperature=0.2, device="cuda", use_relay=USE_RELAY, lora_path=MODEL_PATH)

    return jsonformer()


def output_fn(prediction, accept):
    """
    Post-processing function for model predictions. It gets executed after the predict_fn.
    """

    return json.dumps(prediction), accept