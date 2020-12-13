import os

import torch

ENV_NAME = 'InvertedDoublePendulum-v2'


# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()


def get_default_save_filename(episodes, threads, discount, step_max, actor_lr, critic_lr):
    return f"A3C--{episodes}-{threads}-{str(discount).replace('.', '_')}-{str(step_max).replace('.', '_')}-" \
           f"{str(actor_lr).replace('.', '_')}-{str(critic_lr).replace('.', '_')}"


def ensure_unique_path(path):
    if os.path.exists(path):
        counter = 1
        while os.path.exists(path + f"({str(counter)})"):
            counter += 1
        return path + f"({str(counter)})"
    return path
