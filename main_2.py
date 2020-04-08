import argparse
import random
from math import sqrt, log
from main_1a import MP_method

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='mean', type=int, help="Expectation value or mean")
    parser.add_argument('-v', dest='var', type=int, help="Variance")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()

    print(f'Data point source function: N({float(args.mean)}, {float(args.var)})\n')

    num = 0
    std = sqrt(args.var)
    pre_mean = 0
    pre_var = 0
    mean = 0
    var = 0

    while abs(mean - args.mean) > 0.01 or abs(var - args.var) > 0.01:
        # use Welford's online algorithm
        data_point = MP_method(args.mean, std)

        num += 1
        if num == 1:
            mean = data_point
            var = 0
        else:
            pre_mean = mean
            pre_var = var
            mean = (pre_mean * (num-1) + data_point) / num
            var = pre_var + ((data_point - pre_mean) * (data_point - mean) - pre_var) / num

        print(f"Add data point: {data_point}")
        print(f"Mean = {mean:.15f}   Variance = {var:.15f}\n")
