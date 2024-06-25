import json
from .code.inference import model_fn, predict_fn, input_fn, output_fn

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