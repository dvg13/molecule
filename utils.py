import numpy as np

def get_balanced_class_weights(y):
    n,l = y.shape
    pos = np.sum(y,axis=0)
    neg = n - pos

    return [
        {
            0:1 if pos[i] >= neg[i] else pos[i]/neg[i],
            1:1 if neg[i] >= pos[i] else neg[i]/pos[i],
        }
        for i in range(l)
    ]
