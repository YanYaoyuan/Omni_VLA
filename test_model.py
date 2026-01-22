# inspect_safetensors.py

from safetensors.torch import load_file

MODEL_PATH = "/data/base_model/pi0_base_torch/model.safetensors"

state_dict = load_file(MODEL_PATH, device="cpu")

print(f"Total tensors: {len(state_dict)}\n")

paligemma_state = {
    k.replace("paligemma_with_expert.paligemma.", "", 1): v
    for k, v in state_dict.items()
    if k.startswith("paligemma_with_expert.paligemma.")
}

# 3. 加载
missing, unexpected = model.paligemma.load_state_dict(
    paligemma_state,
    strict=True
)

print("Missing:", missing)
print("Unexpected:", unexpected)