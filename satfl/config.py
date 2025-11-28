import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Dataloading defaults
DEFAULT_BATCH_SIZE = 32
NUM_WORKERS = 2