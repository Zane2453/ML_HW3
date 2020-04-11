import argparse
import random
from math import sqrt, log
import numpy as np
from main_1b import Polynomial
import matplotlib.pyplot as plt

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='pre', type=int, help="The Precision for Initial Prior")
    parser.add_argument('-n', dest='base', type=int, help="Basis Number")
    parser.add_argument('-a', dest='var', type=int, help="The Precision for Likelihood")
    parser.add_argument('-w', dest='weight', type=float, nargs="+", help="Weight Vector")
    args = parser.parse_args()

    return args

def plot_result(stage, result, var, weights, covar):
    if stage == 'ground':
        plt.subplot(2, 2, 1)
        plt.title("Ground truth", fontsize=12)
    elif stage == 'final':
        plt.subplot(2, 2, 2)
        plt.title("Predict result", fontsize=12)
    elif stage == 'ten':
        plt.subplot(2, 2, 3)
        plt.title("After 10 incomes", fontsize=12)
    elif stage == 'fifty':
        plt.subplot(2, 2, 4)
        plt.title("After 50 incomes", fontsize=12)

    plot_x = np.linspace(-2, 2, 100)
    plt.xlim(-2, 2)
    plt.ylim(-20, 25)

    if stage == 'ground':
        plt.plot(plot_x, Polynomial(plot_x, 0, weights), 'black')
        plt.plot(plot_x, Polynomial(plot_x, 0, weights) + var, 'red')
        plt.plot(plot_x, Polynomial(plot_x, 0, weights) - var, 'red')
    else:
        plt.scatter(result[:, 0], result[:, 1])

        plot_y = np.empty((0, 1), float)
        plot_y_var = np.empty((0, 1), float)
        for x in plot_x:
            design_x = np.array([x ** i for i in range(len(weights))])
            plot_y = np.append(plot_y, np.array([[np.dot(design_x, weights)[0]]]), axis=0)
            plot_y_var = np.append(plot_y_var, np.array([[1/var + np.dot(np.dot(design_x, np.linalg.inv(covar)), np.transpose(design_x))]]), axis=0)

        plt.plot(plot_x, plot_y, 'black')
        plt.plot(plot_x, plot_y + plot_y_var, 'red')
        plt.plot(plot_x, plot_y - plot_y_var, 'red')

if __name__ == "__main__":
    args = set_args()

    b = args.pre
    a = args.var
    std = sqrt(a)
    base = args.base
    weights = np.array([weight for weight in args.weight])
    posterior_mean = np.zeros((base, 1))
    posterior_covar = np.identity(base) * b

    data_num = 0
    X = np.empty((0, base), float)
    Y = np.empty((0, 1), float)
    predict = np.empty((0, 2), float)

    plot_result('ground', None, a, weights, None)

    while True:
        data_x = random.uniform(-1, 1)
        data_y = Polynomial(data_x, 0, weights)
        design_x = np.array([data_x ** i for i in range(len(weights))])

        X = np.append(X, np.array([design_x]), axis=0)
        Y = np.append(Y, np.array([[data_y]]), axis=0)
        m = np.copy(posterior_mean)
        S = np.copy(posterior_covar)
        posterior_covar = a * np.dot(np.transpose(X), X) + S
        posterior_mean = np.dot(np.linalg.inv(posterior_covar), a * np.dot(np.transpose(X), Y) + np.dot(S, m))

        predict_Y_mean = np.dot(design_x, posterior_mean)[0]
        predict_Y_var = 1/a + np.dot(np.dot(design_x, np.linalg.inv(posterior_covar)), np.transpose(design_x))
        predict = np.append(predict, np.array([[data_x, predict_Y_mean]]), axis=0)

        print(f"Add data point ({data_x}, {data_y}):\n")
        print(f'Posterior mean:\n{posterior_mean}\n')
        print(f'Posterior variance:\n{np.linalg.inv(posterior_covar)}\n')
        print(f"\nPredictive Distribution ~ N({predict_Y_mean}, {predict_Y_var})\n")
        print("-----------------------")

        data_num += 1

        if data_num == 10:
            plot_result('ten', predict, a, posterior_mean, posterior_covar)
        elif data_num == 50:
            plot_result('fifty', predict, a, posterior_mean, posterior_covar)

        if np.allclose(m, posterior_mean):
            break

    plot_result('final', predict, predict_Y_var, posterior_mean, posterior_covar)
    plt.tight_layout(pad=0.5)
    plt.show()