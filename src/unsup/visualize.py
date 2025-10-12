from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def scatter2d(Z: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "", savepath: Optional[str] = None):
    plt.figure()
    if labels is None:
        plt.scatter(Z[:,0], Z[:,1])
    else:
        plt.scatter(Z[:,0], Z[:,1], c=labels)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
    return plt.gcf()
