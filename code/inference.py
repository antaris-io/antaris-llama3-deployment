"""
Template taken from:
https://towardsdatascience.com/deploy-a-custom-ml-model-as-a-sagemaker-endpoint-6d2540226428
"""
import json
import os
from transformers import  AutoTokenizer
from vllm import LLM
import boto3

# local imports
from .jsonformer import Jsonformer
from .processing import Preprocessor

# Define the S3 bucket and model path
S3_BUCKET = "s3://253333439226-app-registry/llama3"
S3_BASE_MODEL_PATH = "Meta-Llama-3-8B-hf"
S3_MODEL_PATH = "Meta-Llama-3-8B-hf-finetuned"


basepath = os.path.dirname(__file__)
#BASE_MODEL = os.path.join(basepath, "../Meta-Llama-3-8B-hf")
#MODEL_PATH = os.path.join(basepath, "../Meta-Llama-3-8B-hf-finetuned")
BASE_MODEL = "/opt/ml/base_model"
MODEL_PATH = "/opt/ml/fine_tuned_model"
JSON_SCHEME_PATH = os.path.join(basepath, "json_scheme.json")
PROMP_PATH = os.path.join(basepath, "prompt.txt")
MODEL_SIZE = 4*2048
USE_RELAY = True


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)



# ============================================================
# SageMaker Inference Functions
# ============================================================

def download_model_from_s3():
    s3 = boto3.client('s3')
    if not os.path.exists('/opt/ml/model'):
        os.makedirs('/opt/ml/model')
    # Download folders from S3
    s3.download_file(S3_BUCKET, S3_BASE_MODEL_PATH, '/opt/ml/base_model')
    s3.download_file(S3_BUCKET, S3_MODEL_PATH, '/opt/ml/fine_tuned_model')




def model_fn(model_dir):
    """
    This function is the first to get executed upon a prediction request,
    it loads the model from the disk and returns the model object which will be used later for inference.
    """

    # Download model from S3
    download_model_from_s3()

    # Load the model
    model = LLM(BASE_MODEL, enable_lora=True, max_lora_rank=64)

    return model


def input_fn(text, request_content_type):
    """
    The request_body is passed in by SageMaker and the content type is passed in 
    via an HTTP header by the client (or caller).
    """
    # Load raw prompt
    with open(PROMP_PATH, "r") as f:
        prompt_raw = f.read()

    # Load json scheme
    with open(JSON_SCHEME_PATH, "r") as f:
        json_scheme = json.load(f)

    if request_content_type == "text/csv":
        # get the number of tokens
        tokens_offset = len(tokenizer.encode(f"""{prompt_raw.format(text="")} {str(json_scheme)}"""))
        processor = Preprocessor(tokenizer, MODEL_SIZE-2*tokens_offset)

        return processor(text)

    # If the request_content_type is not as expected, raise an exception
    raise ValueError(f"Content type {request_content_type} is not supported")


def predict_fn(prompt, model):
    """
    This function takes in the input data and the model returned by the model_fn
    It gets executed after the model_fn and its output is returned as the API response.
    """

    # Load json scheme
    with open(JSON_SCHEME_PATH, "r") as f:
        json_scheme = json.load(f)

    jsonformer = Jsonformer(model, tokenizer, json_scheme, prompt, temperature=0.2, device="cuda", use_relay=USE_RELAY, lora_path=MODEL_PATH)

    return jsonformer()


def output_fn(prediction, accept):
    """
    Post-processing function for model predictions. It gets executed after the predict_fn.
    """

    return json.dumps(prediction), accept