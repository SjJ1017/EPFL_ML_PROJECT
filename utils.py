import numpy as np

def insert_columns(A, vecs, ks):
    A = np.asarray(A)
    vecs = [np.asarray(v) for v in vecs]
    n, m = A.shape
    k = len(vecs)
    if len(ks) != k:
        raise ValueError("len(vecs) must equal len(ks)")
    final_m = m + k

    if any((pos < 0 or pos >= final_m) for pos in ks):
        raise ValueError(f"each k must be in [0, {final_m-1}]")
    if len(set(ks)) != k:
        raise ValueError("ks contains duplicates (each target column must be unique)")

    res = np.empty((n, final_m), dtype=A.dtype)

    for v, pos in zip(vecs, ks):
        v = v.reshape(n)
        res[:, pos] = v

    src = 0
    for j in range(final_m):
        if j not in ks:
            res[:, j] = A[:, src]
            src += 1

    return res

def replace_columns(A, vecs, ks):
    A = np.asarray(A).copy()
    vecs = [np.asarray(v) for v in vecs]

    n, m = A.shape
    k = len(vecs)

    if len(ks) != k:
        raise ValueError("len(vecs) must equal len(ks)")
    if any(pos < 0 or pos >= m for pos in ks):
        raise ValueError(f"each index in ks must be in [0, {m-1}]")
    if len(set(ks)) != k:
        raise ValueError("ks contains duplicates")

    for v, pos in zip(vecs, ks):
        v = v.reshape(n)
        A[:, pos] = v

    return A