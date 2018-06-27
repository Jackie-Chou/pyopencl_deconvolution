import numpy as np
import os
import sys

if __name__ == "__main__":
    argv = sys.argv
    if not len(argv) == 7:
        raise Exception("argv: h, w, kh, kw, ic, oc")
    h, w, kh, kw, ic, oc = tuple(map(int, argv[1:]))
    feature = np.random.randn(h, w, ic).astype(np.float32)
    kernel = np.random.randn(oc, ic, kh, kw).astype(np.float32)
    np.savez("input.npz", feature=feature, kernel = kernel)
    os.system("python deconv.py -i input.npz -o output.npz -g 1")

