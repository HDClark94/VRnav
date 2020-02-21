# this script will pull relavent features from each recording and compile them into a large dataframe from which you can analyse in whatever way you like.

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
import json
from summarize.common import *
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def process(results_path, researcher):
    results = pd.read_pickle(results_path)

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                         (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                         (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

    elif researcher == "Maja":
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    stops_on_track = plt.figure(figsize=(6, 6))
    #ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fig, ax = plt.subplots()

    distances = np.arange(0,400)
    reward_zone_size =  13.801
    ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)
    if researcher == "Emre":
        cm = plt.cm.get_cmap('viridis')
        dd = ax.scatter(results["target"], results["first_stop_location_relative_to_ip"], c=results["gain_std"], cmap=cm)
        cbar = plt.colorbar(dd)
        cbar.ax.locator_params(nbins=5)
        cbar.set_label('Gain SD', rotation=270, labelpad=12)

    elif researcher == "Maja":
        cm = ListedColormap(['yellow', 'blue'])
        results.loc[results.movement_mechanism == "analogue", 'movement_mechanism'] = 0
        results.loc[results.movement_mechanism == "button_tap", 'movement_mechanism'] = 1

        dd = ax.scatter(results["target"], results["first_stop_location_relative_to_ip"], c=results["movement_mechanism"], cmap=cm)

        yellow_patch = mpatches.Patch(color='yellow', label='Button')
        blue_patch = mpatches.Patch(color='blue', label='Joystick')
        plt.legend(handles=[yellow_patch, blue_patch])

    plt.ylabel('Response (VU)', fontsize=12, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=12, labelpad=10)
    #stops_on_track.colorbar(im, ax=ax)
    plt.xlim(0, max(distances))
    plt.ylim(0, max(distances)+100)

    x = results["target"].values
    y = results["first_stop_location_relative_to_ip"].values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression
    Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
    plt.plot(distances, Y_pred, color='red')

    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')
    plt.savefig(results_path + '/bias.png', dpi=200)
    plt.show()
    plt.close()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre\processed_results.pkl"
    process(results_path, "Emre")
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya\processed_results.pkl"
    process(results_path, "Maja")

if __name__ == '__main__':
    main()