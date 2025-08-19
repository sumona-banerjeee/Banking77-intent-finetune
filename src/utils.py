import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_label_names_from_dataset(dataset):
    return dataset["train"].features["label"].names