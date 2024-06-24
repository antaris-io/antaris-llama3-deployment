import json
from .code import model_fn, predict_fn, input_fn, output_fn

with open("/home/s6jakrei_hpc/antaris-sagemaker-llm/examples/fabucar_input.txt", "r") as f:
    payload = f.read()

response, accept = output_fn(
    predict_fn(
        prompt = input_fn(payload, "text/csv"),
        model = model_fn("../")
    ),
    "text/csv"
)

print("======== OUTPUT ========")
print(json.loads(response))