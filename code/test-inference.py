import json
import sys
import logging
from inference import model_fn, predict_fn, input_fn, output_fn

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Starting inference...")

with open("/home/s6jakrei_hpc/antaris-sagemaker-llm/examples/fabucar_input.txt", "r") as f:
    payload = f.read()

response, accept = output_fn(
    predict_fn(
        prompt = input_fn(payload, "text/csv"),
        model_tokenizer = model_fn("../")
    ),
    "text/csv"
)

print("======== OUTPUT ========")
print(json.loads(response))