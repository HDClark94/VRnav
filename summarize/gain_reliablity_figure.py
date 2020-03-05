# this script will pull relavent features from each recording and compile them into a large dataframe from which you can analyse in whatever way you like.

import numpy as np
from summarize.plotting import *
import json
from summarize.common import *
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches#
import math

def plot_gaussians():
    fig, ax = plt.subplots()

    draw_delta(ax, mu=4)
    draw_guassian(ax, sigma=0.5, mu=4, color="orange")
    draw_guassian(ax, sigma=2, mu=4, color="green")

    plt.ylabel('Probability', fontsize=26, labelpad=15)
    plt.xlabel('Gain', fontsize=26, labelpad=10)
    plt.ylim(0, 1)
    plt.xlim(0, 8)
    ax.set_yticklabels(["0", " ", " ", " ", " "])
    ax.set_xticklabels(["0", " ", " ", " ", "1", " ", " ", " ", "2"])
    ax.tick_params(axis='both', which='major', labelsize=15)

    adjust_spines(ax, ['left','bottom']) # removes top and right spines

    #plt.legend(prop={"size":20})
    plt.title("Trial Gain", fontsize=26)
    plt.show()

def draw_guassian(ax, sigma=1, mu=0, color="orange"):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    x = np.linspace(0,10, 1000)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), label="\u03C3 = "+str(sigma), linewidth=6, color=color)
    return ax

def draw_delta(ax, mu=0):
    xs = np.linspace(-100, 100, 100000)
    delta = np.zeros(len(xs))
    delta[np.abs(np.subtract.outer(xs, mu)).argmin(0)] = 1
    ax.plot(xs, delta,  label="\u03C3 = 0", linewidth=6, color="blue")
    return ax

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    plot_gaussians()

if __name__ == '__main__':
    main()