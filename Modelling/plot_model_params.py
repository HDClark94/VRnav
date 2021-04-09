
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from Modelling.no_kalman_model import *
from Modelling.Stochastic_angelaki import *
from scipy import stats
from scipy.optimize import least_squares
from Modelling.params import Parameters

np.random.seed(64)

def plot_model_parameters(model_params, save_path, trial_type):

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    #plt.title("All subjects", fontsize="20")

    if trial_type == "beaconed":
        colour="k"
    elif trial_type =="non_beaconed":
        colour="r"

    plt.scatter(model_params["gamma"], model_params["lambda"], marker="o", s=50, color=colour)
    plt.xlabel("Gamma", fontsize=20)
    plt.xlim((0,4))
    plt.ylim((0,2))
    plt.ylabel("Lambda", fontsize=20)
    plt.subplots_adjust(left=0.2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.vlines(1, ymin=0, ymax=2, colors='k', linestyles='dashed')
    plt.hlines(1, xmin=0, xmax=4, colors='k', linestyles='dashed')
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.savefig(save_path+"/"+trial_type+"model_parameter_scatter.png",dpi=300)
    plt.show()
    plt.close()

def plot_model_parameters_vectors(beaconed_model_params, non_beaconed_model_params, save_path):
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial


    beaconed_model_params = beaconed_model_params.sort_values(by="ppid")
    non_beaconed_model_params = non_beaconed_model_params.sort_values(by="ppid")

    #plt.scatter(beaconed_model_params["gamma"], beaconed_model_params["lambda"], color="k", marker="o", s=50)
    #plt.scatter(non_beaconed_model_params["gamma"], non_beaconed_model_params["lambda"], color="r", marker="o", s=50)

    for i in range(len(beaconed_model_params)):
        plt.arrow(x=beaconed_model_params["gamma"].iloc[i], y=beaconed_model_params["lambda"].iloc[i],
                  dx=non_beaconed_model_params["gamma"].iloc[i] - beaconed_model_params["gamma"].iloc[i],
                  dy=non_beaconed_model_params["lambda"].iloc[i] - beaconed_model_params["lambda"].iloc[i], color="blue",head_width=0.05)

    plt.xlabel("Gamma (Velocity Prior)", fontsize=20)
    plt.xlim((0,4))
    plt.ylim((0,2))
    plt.ylabel("Lambda (Positional Uncertainty)", fontsize=20)
    plt.subplots_adjust(left=0.2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.vlines(1, ymin=0, ymax=2, colors='k', linestyles='dashed')
    plt.hlines(1, xmin=0, xmax=4, colors='k', linestyles='dashed')
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.savefig(save_path+"/model_parameter_scatter_vector.png", dpi=300)
    plt.show()
    plt.close()


def main():
    print("run something here")

    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model"
    model_params = pd.read_pickle(save_path+"/model_parameters.pkl")

    beaconed_model_params = model_params[model_params["trial_type"] == "beaconed"]
    non_beaconed_model_params = model_params[model_params["trial_type"] =="non_beaconed"]

    plot_model_parameters(beaconed_model_params, save_path, trial_type="beaconed")
    plot_model_parameters(non_beaconed_model_params, save_path, trial_type="non_beaconed")

    plot_model_parameters_vectors(beaconed_model_params, non_beaconed_model_params, save_path)

    #save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\stochastic_angelaki_model"
    #fit_stochastic(human_data, save_path, trial_type="non_beaconed")
    #fit_stochastic(human_data, save_path, trial_type="beaconed")




if __name__ == '__main__':
    main()