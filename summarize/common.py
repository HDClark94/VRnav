import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
from matplotlib.lines import Line2D

def num2str(number):
    # it maybe has a decimal point in
    # and replaces with -
    return str(number).replace(".", "-")


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

        error_from_rz_mid = ((trial_results["Reward Boundary Max"][index] + trial_results["Reward Boundary Min"][index])/2) \
                            - trial_results["first_stop_location_post_cue"][index]

        #error_from_rz_max = trial_results["Reward Boundary Max"][index] - trial_results["first_stop_location_post_cue"][index]
        #error_from_rz_min = trial_results["Reward Boundary Min"][index] - trial_results["first_stop_location_post_cue"][index]
        #error = min(np.array(error_from_rz_max), np.array(error_from_rz_min))

        #first_stop_errors.append(np.round(error, decimals=3))
        first_stop_errors.append(np.round(error_from_rz_mid, decimals=3))

    trial_results["first_stop_post_cue_error"] = first_stop_errors
    trial_results["absolute_first_stop_post_cue_error"] = np.abs(first_stop_errors)

    return trial_results

def extract_first_stop_error(trial_results, session_paths):
    first_stop_errors = []

    for index, _ in trial_results.iterrows():

        error_from_rz_mid = ((trial_results["Reward Boundary Max"][index] + trial_results["Reward Boundary Min"][index])/2)\
                            - trial_results["first_stop_location"][index]

        #error_from_rz_max = trial_results["Reward Boundary Max"][index] - trial_results["first_stop_location"][index]
        #error_from_rz_min = trial_results["Reward Boundary Min"][index] - trial_results["first_stop_location"][index]
        #error = min(np.array(error_from_rz_max), np.array(error_from_rz_min))
        #first_stop_errors.append(np.round(error, decimals=3))

        first_stop_errors.append(np.round(error_from_rz_mid, decimals=3))

    trial_results["first_stop_error"] = first_stop_errors
    trial_results["absolute_first_stop_error"] = np.abs(first_stop_errors)

    return trial_results


def extract_gain(trial_results, baseline_acceleration=4):
    gains = []

    for index, _ in trial_results.iterrows():
        gain = trial_results["Acceleration"][index]/baseline_acceleration
        gains.append(gain)

    trial_results["gain"] = gains

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

def filter_stops(trial_stop_locations, trial_stop_times, trial_results, index):
    # remove stops in black box
    filtered_stops = []
    filtered_stop_times = []

    for i in range(len(trial_stop_locations)):
        if trial_stop_locations[i] > trial_results["Track Start"][index] and trial_stop_locations[i] < trial_results["Track End"][index]:
            filtered_stops.append((trial_stop_locations[i]))
            filtered_stop_times.append((trial_stop_times[i]))

    return np.array(filtered_stops), np.array(filtered_stop_times)

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
    first_stop_time=[]
    first_stops_postcue_time =[]
    stop_times_post_cue = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path+"/"+str(trial_results["player_movement_filename"][index]))
        stop_mask = np.append(False, np.diff(np.array(player_movement["pos_x"]))==0)

        trial_stop_locations = np.array(player_movement["pos_x"])[stop_mask]
        trial_stop_locations, indices = np.unique(trial_stop_locations, return_index=True)  # ignores duplicates

        player_movement = correct_time(player_movement)
        trial_stop_times = np.array(player_movement["time"])[stop_mask]
        trial_stop_times = np.take(trial_stop_times, indices) # ignores duplicates

        trial_stop_locations, trial_stop_times = filter_stops(trial_stop_locations, trial_stop_times, trial_results, index)
        trial_stop_locations_post_cue = trial_stop_locations[trial_stop_locations>trial_results["Cue Boundary Max"][index]]
        trial_stop_post_cue_times = trial_stop_times[trial_stop_locations>trial_results["Cue Boundary Max"][index]]

        if len(trial_stop_locations)<1:
            trial_stop_locations=np.array([np.nan])
            trial_stop_times=np.array([np.nan])
        if len(trial_stop_locations_post_cue)<1:
            trial_stop_locations_post_cue=np.array([np.nan])
            trial_stop_post_cue_times = np.array([np.nan])

        stop_locations_offset_postcue.append(np.round(trial_stop_locations_post_cue - trial_results["Cue Boundary Max"][index], decimals=3))
        first_stops_offset_postcue.append(np.round(trial_stop_locations_post_cue[0] - trial_results["Cue Boundary Max"][index], decimals=3))
        stop_locations_offset.append(np.round(trial_stop_locations - trial_results["Cue Boundary Max"][index], decimals=3))
        first_stops_offset.append(np.round(trial_stop_locations[0] - trial_results["Cue Boundary Max"][index], decimals=3))

        stop_locations_postcue.append(np.round(trial_stop_locations_post_cue, decimals=3))
        first_stops_postcue.append(np.round(trial_stop_locations_post_cue[0], decimals=3))
        stop_locations.append(np.round(trial_stop_locations, decimals=3))
        first_stops.append(np.round(trial_stop_locations[0], decimals=3))

        stop_times.append(np.round(trial_stop_times, decimals=3))
        stop_times_post_cue.append(np.round(trial_stop_post_cue_times, decimals=3))
        first_stop_time.append(np.round(trial_stop_times[0], decimals=3))
        first_stops_postcue_time.append(np.round(trial_stop_post_cue_times[0], decimals=3))

    trial_results["stop_locations"] = stop_locations
    trial_results["first_stop_location"] = first_stops
    trial_results["first_stop_location_post_cue"] = first_stops_postcue
    trial_results["stop_locations_postcue"] = stop_locations_postcue

    trial_results["stop_locations_post_cue_relative_to_ip"] = stop_locations_offset_postcue
    trial_results["first_stops_offset_post_cue_relative_to_ip"] = first_stops_offset_postcue
    trial_results["stop_locations_relative_to_ip"] = stop_locations_offset
    trial_results["first_stop_location_relative_to_ip"] = first_stops_offset

    trial_results["stop_times"] = stop_times
    trial_results["stop_times_post_cue"] = stop_times_post_cue
    trial_results["first_stop_time"] = first_stop_time
    trial_results["first_stops_postcue_time"] = first_stops_postcue_time

    return trial_results

def extract_trial_type_errors(trial_results_trial_type, error_collumn):
    uniques_lengths = np.unique(np.asarray(trial_results_trial_type["integration_length"]))
    tt_mean_errors_for_lengths = []
    tt_std_errors_for_lengths = []

    for i in range(len(uniques_lengths)):
        tt_errors = np.array(trial_results_trial_type[error_collumn])[np.array(trial_results_trial_type["integration_length"])==uniques_lengths[i]]
        if len(tt_errors)<1:
            tt_errors=np.nan
        tt_mean_errors_for_lengths.append(np.nanmean(tt_errors))
        tt_std_errors_for_lengths.append(np.nanstd(tt_errors))

    return tt_mean_errors_for_lengths, tt_std_errors_for_lengths

def extract_trial_duration(trial_results):
    trial_results["trial_duration"] = trial_results['end_time'] - trial_results['start_time'] # calculate duration
    return trial_results

def extract_target(trial_results):
    trial_results["target"] = (trial_results["Reward Boundary Max"]+trial_results['Reward Boundary Min'])/2
    return trial_results

def adjust_track_measurements(trial_results):
    trial_results['Cue Boundary Min'] = trial_results['Cue Boundary Min'] - trial_results["Track Start"]
    trial_results['Cue Boundary Max'] = trial_results['Cue Boundary Max'] - trial_results["Track Start"]
    trial_results['Reward Boundary Min'] = trial_results['Reward Boundary Min'] - trial_results["Track Start"]
    trial_results['Reward Boundary Max'] = trial_results['Reward Boundary Max'] - trial_results["Track Start"]
    trial_results['Track End'] = trial_results['Track End'] - trial_results["Track Start"]
    trial_results['Rewarded Location'] = trial_results['Rewarded Location'] - trial_results["Track Start"]
    trial_results['Teleport from'] = trial_results['Teleport from'] - trial_results["Track Start"]
    trial_results['Teleport to'] = trial_results['Teleport to'] - trial_results["Track Start"]
    trial_results['stop_locations'] = trial_results['stop_locations'] - trial_results["Track Start"]
    trial_results['first_stop_location'] = trial_results['first_stop_location'] - trial_results["Track Start"]
    trial_results['first_stop_location_post_cue'] = trial_results['first_stop_location_post_cue'] - trial_results["Track Start"]
    trial_results['stop_locations_postcue'] = trial_results['stop_locations_postcue'] - trial_results["Track Start"]
    trial_results['target'] = trial_results['target'] - trial_results['Track Start']
    trial_results["Track Start"] = trial_results["Track Start"] - trial_results["Track Start"]

    return trial_results

def extract_summary(trial_results, session_path):
    trial_results = split_stop_data_by_block(trial_results, block=2)  # only use block 2, this ejects habituation block 1
    trial_results = extract_stops(trial_results, session_path) # add stop times and locations to dataframe
    trial_results = extract_intergration_distance(trial_results, session_path)
    trial_results = extract_first_stop_error(trial_results,session_path)
    trial_results = extract_first_stop_post_cue_error(trial_results, session_path)
    trial_results = extract_speeds(trial_results, session_path) # adds speeds to dataframe
    trial_results = extract_trial_duration(trial_results)
    trial_results = extract_target(trial_results)
    trial_results = extract_gain(trial_results, baseline_acceleration=4)
    trial_results = adjust_track_measurements(trial_results)

    return trial_results


def calculate_correct(results):
    df = pd.DataFrame()

    unique_experiments = np.unique(results['experiment'])
    for experiment in unique_experiments:
        experiment_results = results[results['experiment'] == experiment]
        unique_participants = np.unique(results['ppid'])
        for id in unique_participants:
            participant_results = experiment_results[experiment_results['ppid'] == id]
            unique_sessions = np.unique(participant_results['session_num'])
            for session in unique_sessions:
                session_results = participant_results[participant_results['session_num'] == session]
                unique_trial_types = np.unique(session_results["Trial type"])
                for trial_type in unique_trial_types:
                    trial_type_results = session_results[session_results['Trial type'] == trial_type]
                    unique_movement_mechs = np.unique(trial_type_results["movement_mechanism"])
                    for movement_mech in unique_movement_mechs:
                        movement_mech_results = trial_type_results[trial_type_results["movement_mechanism"] == movement_mech]
                        unique_track_lengths = np.unique(movement_mech_results["integration_length"])
                        for track_length in unique_track_lengths:
                            track_length_results = movement_mech_results[movement_mech_results["integration_length"] == track_length]

                            df = df.append(pd.DataFrame([{'experiment': experiment,
                                                          'ppid': id,
                                                          'session': int(session),
                                                          'trial_type': trial_type,
                                                          'gain sd': track_length_results["gain_std"].iloc[0], # bodgy but gain sd is always the same within a session
                                                          'distance': track_length,
                                                          'n_trial': len(track_length_results),
                                                          'n_correct': sum(track_length_results["Trial Scored"]),
                                                          'correct_proportion': sum(track_length_results["Trial Scored"])/len(track_length_results)}]))

    return df
