import numpy as np
from summarize.plot_behaviour_summary import *

def translate_to_legacy_format(session_path, scale=True):
    print(" I am making a legacy array for ", session_path)

    legacy_array = []

    trial_results = pd.read_csv(session_path + "/trial_results.csv")
    #trial_results = split_stop_data_by_block(trial_results, block=2)  # only use block 2, this ejects habituation block 1
    trial_results = extract_speeds_same_length(trial_results, session_path)  # adds speeds to dataframe

    track_start = np.asarray(trial_results["Track Start"])
    track_end = np.asarray(trial_results["Track End"])
    reward_start = np.asarray(trial_results["Reward Boundary Min"])
    reward_end =   np.asarray(trial_results["Reward Boundary Max"])
    cue_start = np.asarray(trial_results["Cue Boundary Max"])
    cue_end = np.asarray(trial_results["Cue Boundary Max"])

    #teleport_tos = np.asarray(trial_results["Teleport to"])
    #telport_froms = np.asarray(trial_results["Teleport from"])

    for index, _ in trial_results.iterrows():

        # for reward signal replication
        already_rewarded = False

        player_movement = pd.read_csv(session_path+"/"+str(trial_results["player_movement_filename"][index]))

        ## this is for scaling xpos values to n virtual units
        min_xpos = trial_results["Teleport to"][index]
        max_xpos = trial_results["Teleport from"][index]
        vu_range = [0, 20] # 0-20 virtual units

        for m_index, _ in player_movement.iterrows():

            reward, already_rewarded = translate_reward(trial_results["Rewarded Location"][index], player_movement["pos_x"][m_index], already_rewarded)

            legacy_array.append([player_movement["time"][m_index],                                        # 0: time from session start
                                 player_movement["pos_x"][m_index],                                       # 1: location (suppose to be 0-20)
                                 trial_results["speeds"][index][m_index],                                 # 2: speed cm/s
                                 trial_results["speeds"][index][m_index],                                 # 3: speed cm/s
                                 reward,                                                                  # 4: rewarded (0/1)
                                 0,                                                                       # 5: ?
                                 0,                                                                       # 6: ?
                                 reward,                                                                  # 7: rewarded (0/1)
                                 convert_to_legacy_trialtype(trial_results["Trial type"][index]),         # 8: trialtype (0,10,20) = (b, nb, p)
                                 trial_results["trial_num"][index],                                       # 9: trial number
                                 1,                                                                       # 10: ?
                                 trial_results["Reward Boundary Min"][index],                             # 11: rz start
                                 trial_results["Reward Boundary Max"][index]])                            # 12: rz end

    legacy_array = np.asarray(legacy_array)

    if scale:
        legacy_array[:, 1] =  scale_vu(legacy_array[:, 1],  vu_range, min_xpos, max_xpos)  # location
        legacy_array[:, 2] =  scale_vu_speed(legacy_array[:, 2], vu_range, min_xpos, max_xpos)  # location
        legacy_array[:, 3] =  scale_vu_speed(legacy_array[:, 3], vu_range, min_xpos, max_xpos)  # location
        legacy_array[:, 11] = scale_vu(legacy_array[:, 11], vu_range, min_xpos, max_xpos)  # rz start
        legacy_array[:, 12] = scale_vu(legacy_array[:, 12], vu_range, min_xpos, max_xpos)  # rz end

        track_start = scale_vu(track_start, vu_range, min_xpos, max_xpos)
        track_end =   scale_vu(track_end,   vu_range, min_xpos, max_xpos)

        reward_start = scale_vu(reward_start, vu_range, min_xpos, max_xpos)
        reward_end = scale_vu(reward_end, vu_range, min_xpos, max_xpos)

        cue_start = scale_vu(cue_start, vu_range, min_xpos, max_xpos)
        cue_end = scale_vu(cue_end, vu_range, min_xpos, max_xpos)


    return legacy_array, track_start, track_end, reward_start, reward_end, cue_start, cue_end


# ==================================================================================================================================#

# inhouse functions for datatype translation

def scale_vu_speed(current_speed, range, min, max):
    scaled = (current_speed/(max-min))*(range[1]-range[0])
    return scaled

def scale_vu(current_xpos, range, min, max):    # takes single trial data as input
    scaled = ((range[1]-range[0])*((current_xpos-min)/(max-min)))+range[0]
    # min max scale to within range[0] and range[1]
    return scaled

def translate_reward(rewarded_location, current_location, already_rewarded):

    if not already_rewarded: # eg already = False
        if rewarded_location == current_location:
            return 1, True
        else:
            return 0, False
    else: # eg. already = True
        return 0, True

def convert_to_legacy_trialtype(trial_type_string):

    if trial_type_string == "beaconed":
        return 0
    elif trial_type_string == "non_beaconed":
        return 10
    elif trial_type_string == "probe":
        return 20

# ==================================================================================================================================#