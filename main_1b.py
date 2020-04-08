import argparse
import random
from math import sqrt, log
import numpy as np
from main_1a import MP_method

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='base', type=int, help="Basis Number")
    parser.add_argument('-v', dest='var', type=int, help="Variance")
    parser.add_argument('-w', dest='weight', type=float, nargs="+", help="Weight Vector")
    args = parser.parse_args()

    return args

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