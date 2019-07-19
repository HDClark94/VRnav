import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *


def split_stop_data_by_trial_type(session_dataframe):
    '''
    :param session_dataframe:
    :return: dataframe with selected trial types
    '''

    beaconed = session_dataframe[session_dataframe['Trial type'] == "beaconed"]
    non_beaconed = session_dataframe[session_dataframe['Trial type'] == "non_beaconed"]
    probe = session_dataframe[session_dataframe['Trial type'] == "probe"]
    return beaconed, non_beaconed, probe

def split_stop_data_by_block(session_dataframe, block):
    return session_dataframe[session_dataframe["block_num"] == block]


def correct_time(trial_dataframe):
    # time always starts at zero each trial
    trial_dataframe["time"] = trial_dataframe["time"]-trial_dataframe["time"][0]
    return trial_dataframe

def extract_speeds_same_length(trial_results, session_path):
    speeds = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path + "/" + str(trial_results["player_movement_filename"][index]))

        location_diff = np.diff(np.array(player_movement["pos_x"]))
        time_diff = np.diff(np.array(player_movement["time"]))

        speed = location_diff / time_diff  # this is speed in vu per second
        speed = np.concatenate([[speed[0]], speed])
        speeds.append(speed)

    trial_results["speeds"] = speeds

    # returns speeds with same length as movement steps
    return trial_results


def extract_speeds(trial_results, session_path):
    speeds = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path + "/" + str(trial_results["player_movement_filename"][index]))

        location_diff = np.diff(np.array(player_movement["pos_x"]))*-1
        time_diff = np.diff(np.array(player_movement["time"]))

        speed = location_diff/time_diff # this is speed in vu per second
        speeds.append(speed)

    trial_results["speeds"] = speeds

    return trial_results

def extract_speed_by_spatial_bins(trial_results, session_path):
   # TODO
   return trial_results


def extract_speed_by_time_bins(trial_results, session_path):
    #TODO
    return trial_results

def extract_stops(trial_results, session_path):

    stop_locations = []
    stop_times = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path+"/"+str(trial_results["player_movement_filename"][index]))
        stop_idx = np.where(np.diff(np.array(player_movement["pos_x"]))==0)

        trial_stop_locations = np.array(player_movement["pos_x"])[stop_idx]
        trial_stop_locations = np.unique(trial_stop_locations)  # ignores duplicates

        player_movement = correct_time(player_movement)
        trial_stop_times = np.array(player_movement["time"])[stop_idx]

        stop_locations.append(trial_stop_locations)
        stop_times.append(trial_stop_times)

    trial_results["stop_locations"] = stop_locations
    trial_results["stop_times"] = stop_times

    return trial_results

