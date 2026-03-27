import torch

torch.cuda.synchronize()
results = benchmark.Timer(
    stmt="quantized_model(input_data)",
    globals={
        "quantized_model": quantized_model,
        "input_data": input_data
    }
).blocked_autorange()
torch.cuda.synchronize()