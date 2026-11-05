from __future__ import annotations

import datetime as _dt
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # protocol 4 loads on any python >= 3.4, so the lab box and Streamlit Cloud are both fine
        pickle.dump(obj, f, protocol=4)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# json can't handle numpy ints/floats/arrays (or NaN), so turn everything into plain python first
def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        # NaN isn't valid json -> store it as null
        return None if np.isnan(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_plain(obj), f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# python / numpy versions + host name + time, printed in the notebook as the lab-run proof
def env_info():
    import pandas as pd
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
