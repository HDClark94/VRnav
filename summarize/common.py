import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
from matplotlib.lines import Line2D


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


def extract_intergration_distance(trial_results, session_paths):
    integration_distances = []

    for index, _ in trial_results.iterrows():
        integration_length = np.round(np.array(trial_results["Reward Boundary Min"][index])-trial_results["Cue Boundary Max"][index], decimals=1)

        integration_distances.append(integration_length)

    trial_results["integration_length"] = integration_distances

    return trial_results

def extract_first_stop_post_cue_error(trial_results, session_paths):
    first_stop_errors = []

    for index, _ in trial_results.iterrows():
        error_from_rz_max = trial_results["Reward Boundary Max"][index] - trial_results["first_stop_location_post_cue"][index]
        error_from_rz_min = trial_results["Reward Boundary Min"][index] - trial_results["first_stop_location_post_cue"][index]
        error = min(np.array(error_from_rz_max), np.array(error_from_rz_min))

        first_stop_errors.append(np.round(error, decimals=3))

    trial_results["first_stop_post_cue_error"] = first_stop_errors
    trial_results["absolute_first_stop_post_cue_error"] = np.abs(first_stop_errors)

    return trial_results

def extract_first_stop_error(trial_results, session_paths):
    first_stop_errors = []

    for index, _ in trial_results.iterrows():
        error_from_rz_max = trial_results["Reward Boundary Max"][index] - trial_results["first_stop_location"][index]
        error_from_rz_min = trial_results["Reward Boundary Min"][index] - trial_results["first_stop_location"][index]
        error = min(np.array(error_from_rz_max), np.array(error_from_rz_min))

        first_stop_errors.append(np.round(error, decimals=3))

    trial_results["first_stop_error"] = first_stop_errors
    trial_results["absolute_first_stop_error"] = np.abs(first_stop_errors)

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

def filter_stops(trial_stop_locations, trial_results, index):
    # remove stops in black box
    filtered_stops = []

    for i in range(len(trial_stop_locations)):
        if trial_stop_locations[i] > trial_results["Track Start"][index] and trial_stop_locations[i] < trial_results["Track End"][index]:
            filtered_stops.append((trial_stop_locations[i]))

    return filtered_stops

def extract_stops(trial_results, session_path):

    stop_locations = []
    first_stops = []
    stop_times = []
    stop_locations_offset = []
    first_stops_offset = []
    stop_locations_offset_postcue = []
    first_stops_offset_postcue = []
    stop_locations_postcue =[]
    first_stops_postcue=[]

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path+"/"+str(trial_results["player_movement_filename"][index]))
        stop_idx = np.where(np.diff(np.array(player_movement["pos_x"]))==0)

        trial_stop_locations = np.array(player_movement["pos_x"])[stop_idx]
        trial_stop_locations = np.unique(trial_stop_locations)  # ignores duplicates

        player_movement = correct_time(player_movement)
        trial_stop_times = np.array(player_movement["time"])[stop_idx]

        trial_stop_locations = filter_stops(trial_stop_locations, trial_results, index)
        trial_stop_locations_post_cue = np.array(trial_stop_locations)[trial_stop_locations>trial_results["Cue Boundary Max"][index]]

        if len(trial_stop_locations)<1:
            trial_stop_locations=np.array([np.nan])
        if len(trial_stop_locations_post_cue)<1:
            trial_stop_locations_post_cue=np.array([np.nan])

        stop_locations_offset_postcue.append(np.round(trial_stop_locations_post_cue - trial_results["Cue Boundary Max"][index], decimals=3))
        first_stops_offset_postcue.append(np.round(trial_stop_locations_post_cue[0] - trial_results["Cue Boundary Max"][index], decimals=3))
        stop_locations_offset.append(np.round(trial_stop_locations - trial_results["Cue Boundary Max"][index], decimals=3))
        first_stops_offset.append(np.round(trial_stop_locations[0] - trial_results["Cue Boundary Max"][index], decimals=3))

        stop_locations_postcue.append(np.round(trial_stop_locations_post_cue, decimals=3))
        first_stops_postcue.append(np.round(trial_stop_locations_post_cue[0], decimals=3))
        stop_locations.append(np.round(trial_stop_locations, decimals=3))
        first_stops.append(np.round(trial_stop_locations[0], decimals=3))

        stop_times.append(np.round(trial_stop_times, decimals=3))


    trial_results["stop_locations"] = stop_locations
    trial_results["first_stop_location"] = first_stops
    trial_results["first_stop_location_post_cue"] = first_stops_postcue
    trial_results["stop_locations_postcue"] = stop_locations_postcue

    trial_results["stop_locations_post_cue_relative_to_ip"] = stop_locations_offset_postcue
    trial_results["first_stops_offset_post_cue_relative_to_ip"] = first_stops_offset_postcue
    trial_results["stop_locations_relative_to_ip"] = stop_locations_offset
    trial_results["first_stop_location_relative_to_ip"] = first_stops_offset

    trial_results["stop_times"] = stop_times

    return trial_results

def get_standard_deviation_from_histogram(x, z, bins, means):
    #TODO fix this, its rubbish
    # deviation is same shape as errors
    sample_variances = x.copy()
    for i in range(len(x)):
        tmp = np.digitize(x[i], bins)-1
        if tmp == len(bins)-1:
            tmp = tmp-1

        mean = means[tmp]
        variance = np.square(x[i]-mean)
        sample_variances[i] = variance
    variance = div0(np.histogram(z, bins, weights=sample_variances)[0], np.histogram(z, bins)[0])

    return np.sqrt(variance)

def div0(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
