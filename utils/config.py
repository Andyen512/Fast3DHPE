import os, yaml

def load_cfg(path: str):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
