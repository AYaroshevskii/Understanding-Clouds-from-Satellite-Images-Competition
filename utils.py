from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm_notebook
from torch.utils import data
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import imread
import torch


# Save and load model checkpoint
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {"state_dict": model.state_dict()}
    if optimizer:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, checkpoint_path)
    print("model saved to %s" % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["state_dict"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer"])
    print("model loaded from %s" % checkpoint_path)
