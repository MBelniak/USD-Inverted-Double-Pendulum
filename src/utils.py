import torch

ENV_NAME = 'InvertedDoublePendulum-v2'

# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()
