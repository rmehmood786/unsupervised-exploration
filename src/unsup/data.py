from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
from sklearn import datasets

@dataclass
class Dataset:
    X: pd.DataFrame
    y: Optional[pd.Series] = None
    name: str = "dataset"

def load_builtin(name: str) -> Dataset:
    name = name.lower()
    if name == "iris":
        bunch = datasets.load_iris(as_frame=True)
    elif name == "wine":
        bunch = datasets.load_wine(as_frame=True)
    elif name == "digits":
        bunch = datasets.load_digits(as_frame=True)
    elif name in {"breast_cancer","cancer"}:
        bunch = datasets.load_breast_cancer(as_frame=True)
    else:
        raise ValueError(f"Unknown builtin dataset: {name}")
    return Dataset(X=bunch.data, y=bunch.target, name=name)
