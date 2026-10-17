import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
import datasets

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)

print("\nGPU availability:")
print("CUDA:", torch.cuda.is_available())
print("Apple MPS:", torch.backends.mps.is_available())