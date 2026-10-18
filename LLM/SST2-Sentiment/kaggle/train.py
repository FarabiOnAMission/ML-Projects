import torch
from datasets import load_dataset

# Check GPU availability
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load SST-2
dataset = load_dataset("stanfordnlp/sst2")

print(dataset)