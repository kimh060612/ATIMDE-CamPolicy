import numpy as np

V_STOP = 0.03
W_STOP = 0.05
W_ROTATE = 0.5
W_SPIN = 1.0
A_SLOW = 0.5

MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0


def assign_motion_label(
    v,
    a,
    w,
    v_stop=V_STOP,
    w_stop=W_STOP,
    w_rotate=W_ROTATE,
    w_spin=W_SPIN,
):
    if np.isnan(v):
        v = 0.0
    if np.isnan(a):
        a = 0.0
    if np.isnan(w):
        w = 0.0
    if v < v_stop and w < w_stop:
        return "stop"
    if w >= w_spin and v < 0.25:
        return "spin"
    if w >= w_rotate and v < 0.30:
        return "rotate"
    if a >= A_SLOW and w < w_rotate:
        return "fast"
    if w < w_rotate:
        return "slow"
    if w >= w_rotate:
        return "rotate"
    return "stop"