# this script will pull relavent features from each recording and compile them into a large dataframe from which you can analyse in whatever way you like.

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
import json
import sys
import traceback
from summarize.common import *
from sklearn.linear_model import LinearRegression


def get_trial_pairs(processed_results_df, collumn):
    trial_pairs_df= pd.DataFrame()

    for trial_paired_id in np.unique(processed_results_df["trial_pairs_id"]):
        trial_pair = processed_results_df[(processed_results_df["trial_pairs_id"] == trial_paired_id)]

        #only add trial pairs when the pair has 2 entries and is thus a valid pair
        if len(trial_pair) == 2:
            pair_df_tmp=pd.DataFrame()
            beaconed_trial = trial_pair[(trial_pair["Trial type"] =="beaconed")]
            non_beaconed_trial = trial_pair[(trial_pair["Trial type"] =="non_beaconed")]

            beaconed_value = beaconed_trial[collumn].iloc[0]
            non_beaconed_value = non_beaconed_trial[collumn].iloc[0]
            gain_std = beaconed_trial["gain_std"].iloc[0]

            if not (np.isnan(beaconed_value)) and not (np.isnan(non_beaconed_value)) and \
                    (beaconed_value < 200) and (non_beaconed_value < 200):
                pair_df_tmp["beaconed_"+collumn] = [beaconed_value]
                pair_df_tmp["non_beaconed_"+collumn] = [non_beaconed_value]
                pair_df_tmp["gain_std"] = [gain_std]

            trial_pairs_df = pd.concat([trial_pairs_df, pair_df_tmp], ignore_index=True)

    return trial_pairs_df


def plot_regression(ax, x, y, c, text_x):

    if len(x)<2:
        print("not enough points to plot a regression line")
        return np.nan, np.nan

    # x  and y are pandas collumn
    x = x.values
    y = y.values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)

    pearson_r = stats.pearsonr(x.flatten(),y.flatten())

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression

    x_test = np.linspace(min(x), max(x), 100)

    Y_pred = linear_regressor.predict(x_test.reshape(-1, 1))  # make predictions

    ax.text(  # position text relative to Axes
        text_x, 0.95, "R= "+str(np.round(pearson_r[0], decimals=2)),
        ha='right', va='top',
        transform=ax.transAxes, fontsize=15, color=c)

    ax.plot(x_test, Y_pred, c=c)

    gradient = (Y_pred[-1]-Y_pred[0])/\
               (x_test[-1]-x_test[0])

    return pearson_r[0], gradient[0]

def time_analysis(processed_results_df, save_path):
    '''
    Time analysis looks at the dependence of time in the stopping stradgy in non beaconed trials in beaconed/non beaconed pairs

    We first look at the trial gain in the beaconed trial

    :param processed_results_df: dataframe created from proccess() in concatenate_data.py
    :return: processed_results_df is added collumns
    '''

    time_analysis_df = pd.DataFrame()

    collumn ="first_stop_time"
    for ppid in np.unique(processed_results_df["ppid"]):
        ppid_time_analysis_df = pd.DataFrame()


        ppid_processed_results_df = processed_results_df[(processed_results_df["ppid"] == ppid)]

        ppid_trial_pairs = get_trial_pairs(ppid_processed_results_df, collumn=collumn)

        no_uncertainty = ppid_trial_pairs[(ppid_trial_pairs["gain_std"] == 0)]
        low_uncertainty = ppid_trial_pairs[(ppid_trial_pairs["gain_std"] == 0.5)]
        high_uncertainty = ppid_trial_pairs[(ppid_trial_pairs["gain_std"] == 2)]


        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1) #stops per trial
        plt.title(ppid, fontsize="20")
        plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")

        plt.scatter(no_uncertainty["beaconed_"+collumn], no_uncertainty["non_beaconed_"+collumn], color="blue", marker="x")
        plt.scatter(low_uncertainty["beaconed_"+collumn], low_uncertainty["non_beaconed_"+collumn], color="orange", marker="x")
        plt.scatter(high_uncertainty["beaconed_"+collumn], high_uncertainty["non_beaconed_"+collumn], color="green", marker="x")

        pearson_r_noUncertainty, gradient_noUncertainty = plot_regression(ax, no_uncertainty["beaconed_"+collumn], no_uncertainty["non_beaconed_"+collumn], c="blue", text_x=0.3)
        pearson_r_lowUncertainty, gradient_lowUncertainty = plot_regression(ax, low_uncertainty["beaconed_"+collumn], low_uncertainty["non_beaconed_"+collumn], c="orange", text_x=0.55)
        pearson_r_highUncertainty, gradient_highUncertainty = plot_regression(ax, high_uncertainty["beaconed_"+collumn], high_uncertainty["non_beaconed_"+collumn], c="green", text_x=0.8)

        max_t = max([max(ppid_trial_pairs["beaconed_"+collumn]), max(ppid_trial_pairs["non_beaconed_"+collumn])])

        plt.xlabel("Beaconed Stop Time", fontsize=20)
        plt.xlim((0, max_t))
        plt.ylim((0,max_t))
        plt.locator_params(axis='y', nbins=5)
        plt.ylabel("Non Beaconed Stop Time", fontsize=20)
        plt.subplots_adjust(left=0.2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        #plt.legend(loc="upper left")
        plt.savefig(save_path+"/"+ppid+"_time_analysis.png")
        plt.show()
        plt.close()

        ppid_time_analysis_df["ppid"] = [ppid]
        ppid_time_analysis_df["pearson_r_noUncertainty"] = [pearson_r_noUncertainty]
        ppid_time_analysis_df["pearson_r_lowUncertainty"] = [pearson_r_lowUncertainty]
        ppid_time_analysis_df["pearson_r_highUncertainty"] = [pearson_r_highUncertainty]
        ppid_time_analysis_df["gradient_noUncertainty"] = [gradient_noUncertainty]
        ppid_time_analysis_df["gradient_lowUncertainty"] = [gradient_lowUncertainty]
        ppid_time_analysis_df["gradient_highUncertainty"] = [gradient_highUncertainty]

        time_analysis_df = pd.concat([time_analysis_df, ppid_time_analysis_df], ignore_index=True)

    return time_analysis_df

def plot_r_comparison(time_analysis_df, save_path):

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    #plt.title(ppid, fontsize="20")

    objects = ["None", "Low", "High"]
    objects_collumns = ["pearson_r_noUncertainty", "pearson_r_lowUncertainty", "pearson_r_highUncertainty"]
    objects_colors = ["blue", "orange", "green"]
    x_pos = np.arange(len(objects))

    for i in range(len(objects)):
        objects_collumn = objects_collumns[i]
        objects_color = objects_colors[i]

        ax.scatter(x_pos[i]*np.ones(len(np.asarray(time_analysis_df[objects_collumn]))), np.asarray(time_analysis_df[objects_collumn]), edgecolor=objects_color, marker="o", facecolors='none')
        ax.errorbar(x_pos[i], np.nanmean(np.asarray(time_analysis_df[objects_collumn])), yerr=stats.sem(np.asarray(time_analysis_df[objects_collumn]), nan_policy="omit"), ecolor=objects_color, capsize=10, fmt="o", color=objects_color)


        print(objects_collumn)
        print("====================================")
        ordered = time_analysis_df.sort_values(by=objects_collumn)
        print(np.asarray(ordered[objects_collumn]))
        print(np.asarray(ordered["ppid"]))
        print("++++++++++++++++++++++++++++++++++++")


    plt.xlabel("T-by-T Gain Standard Deviation", fontsize=20)
    plt.xlim((-1, 3))
    plt.ylim((-0.5, 1))
    plt.xticks(x_pos, objects, fontsize=8)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.ylabel("Pearson R", fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.legend(loc="upper left")
    plt.savefig(save_path+"/Pearson_r_comparison.png")
    plt.show()
    plt.close()

def plot_gradient_comparison(time_analysis_df, save_path):

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    #plt.title(ppid, fontsize="20")

    objects = ["None", "Low", "High"]
    objects_collumns = ["gradient_noUncertainty", "gradient_lowUncertainty", "gradient_highUncertainty"]
    objects_colors = ["blue", "orange", "green"]
    x_pos = np.arange(len(objects))

    for i in range(len(objects)):
        objects_collumn = objects_collumns[i]
        objects_color = objects_colors[i]

        ax.scatter(x_pos[i]*np.ones(len(np.asarray(time_analysis_df[objects_collumn]))), np.asarray(time_analysis_df[objects_collumn]), edgecolor=objects_color, marker="o", facecolors='none')
        ax.errorbar(x_pos[i], np.nanmean(np.asarray(time_analysis_df[objects_collumn])), yerr=stats.sem(np.asarray(time_analysis_df[objects_collumn]), nan_policy="omit"), ecolor=objects_color, capsize=10, fmt="o", color=objects_color)

        print(objects_collumn)
        print("====================================")
        ordered = time_analysis_df.sort_values(by=objects_collumn)
        print(np.asarray(ordered[objects_collumn]))
        print(np.asarray(ordered["ppid"]))
        print("++++++++++++++++++++++++++++++++++++")


    plt.xlabel("T-by-T Gain Standard Deviation", fontsize=20)
    plt.xlim((-1, 3))
    plt.ylim((-1, 1.5))
    plt.xticks(x_pos, objects, fontsize=8)
    #plt.xticks(rotation=-45)
    plt.locator_params(axis='y', nbins=5)
    plt.ylabel("Gradient", fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.legend(loc="upper left")
    plt.savefig(save_path+"/gradient_comparison.png")
    plt.show()
    plt.close()

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    emre_processed_results = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\processed_results.pkl")
    maja_processed_results = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Maya\processed_results.pkl")

    time_analysis_emre = time_analysis(emre_processed_results, save_path=r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\figs\time_analysis")
    #time_analysis_maja = time_analysis(maja_processed_results, save_path=r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Maya\figs\time_analysis")

    plot_r_comparison(time_analysis_emre, save_path=r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\figs\time_analysis")
    plot_gradient_comparison(time_analysis_emre, save_path=r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\figs\time_analysis")

if __name__ == '__main__':
    main()