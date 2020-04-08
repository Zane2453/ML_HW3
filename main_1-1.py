import argparse
import random
from math import sqrt, log

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mean', type=int, help="Expectation value or mean")
    parser.add_argument('-v', dest='var', type=int, help="Variance")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()

    # use Marsaglia polar method
    while True:
        u = random.uniform(-1, 1)
        v = random.uniform(-1, 1)
        if (u**2) + (v**2) < 1:
            break;

    s = (u**2) + (v**2)
    x = u * sqrt((-2) * log(s) / s)
    y = v * sqrt((-2) * log(s) / s)

    mean = args.mean
    std = sqrt(args.var)
    data_point = mean + (std * x)
    print(f"Normal Distribution Data point: {data_point}")