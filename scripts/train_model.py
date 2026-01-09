import os, sys, yaml
from chessbot import MODEL_DIR, SP_DIR

import numpy as np
from xerces_training.chunkparser import ChunkParser

from chessbot import MODEL_DIR, SP_DIR
from chessbot.utils import batch_policy_metrics, print_validation, format_time
from chessbot.review import GameViewer, load_game_index, ANALYZE_PKL
from chessbot.model import load_model, save_model

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def load_training_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def run_training(cfg):
    
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    run_dir = os.path.join(SP_DIR, run_tag)

    if not os.path.isdir(run_dir):
        print(f"[error] run dir not found: {run_dir}")
        sys.exit(1)

    # find run config yaml
    yaml_path = None
    for nm in ("training_config.yaml", "training_config.yml"):
        p = os.path.join(run_dir, nm)
        if os.path.exists(p):
            yaml_path = p
            break
        
    if yaml_path is None:
        print(f"[error] no training_config.yaml found in {run_dir}")
        sys.exit(1)

    # load base config and validation configs
    cfg = load_training_config(yaml_path)
    
    # check for the model
    model_file = run_tag + "_model.h5"
    model_path = os.path.join(run_dir, model_file)
    if not os.path.exists(model_path):
        print(f"[error] no {model_file} found in {run_dir}")
        sys.exit(1)
        
    model = run_training(cfg)
        
    

