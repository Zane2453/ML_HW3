import argparse
import random
from math import sqrt, log
import numpy as np

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='base', type=int, help="Basis Number")
    parser.add_argument('-v', dest='var', type=int, help="Variance")
    parser.add_argument('-w', dest='weight', type=float, nargs="+", help="Weight Vector")
    args = parser.parse_args()

    return args

# use Marsaglia polar method
def MP_method(mean, std):
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        if (u**2) + (v**2) < 1:
            break;

    s = (u**2) + (v**2)
    x = u * sqrt((-2) * log(s) / s)
    y = v * sqrt((-2) * log(s) / s)

    return mean + (std * x)

if __name__ == "__main__":
    args = set_args()

    mean = 0
    std = sqrt(args.var)
    base = args.base
    weights = np.array([weight for weight in args.weight])
    x = random.uniform(-1, 1)
    e = MP_method(mean, std)

    y = e
    for index, weight in enumerate(weights):
        y += (x**index) * weight

    print(f'Polynomial Basis value: {y}')