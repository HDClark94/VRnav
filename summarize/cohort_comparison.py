import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
from summarize.common import *

def get_compact_cohort_error_std_results(cohort, error_collumn):
    cohort_results = pd.DataFrame()

    cohort_b_errors = []
    cohort_nb_errors = []
    cohort_p_errors = []
    cohort_b_stds = []
    cohort_nb_stds = []
    cohort_p_stds = []

    for session_path in cohort:
        trial_results = pd.read_csv(session_path+"/trial_results.csv")
        trial_results = extract_summary(trial_results, session_path)

        beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)

        b_errors, b_stds = extract_trial_type_errors(beaconed, error_collumn)
        nb_errors, nb_stds = extract_trial_type_errors(non_beaconed, error_collumn)
        p_errors, p_stds = extract_trial_type_errors(probe, error_collumn)

        cohort_b_errors.append(b_errors)
        cohort_nb_errors.append(nb_errors)
        cohort_p_errors.append(p_errors)
        cohort_b_stds.append(b_stds)
        cohort_nb_stds.append(nb_stds)
        cohort_p_stds.append(p_stds)

    cohort_results["beaconed_errors_for_unique_length"] = cohort_b_errors
    cohort_results["non_beaconed_errors_for_unique_length"] = cohort_nb_errors
    cohort_results["probe_errors_for_unique_length"] = cohort_p_errors
    cohort_results["beaconed_stds_for_unique_length"] = cohort_b_stds
    cohort_results["non_beaconed_stds_for_unique_length"] = cohort_nb_stds
    cohort_results["beaconed_errors_for_unique_length"] = cohort_p_stds

    return cohort_results

def plot_cohort_comparison_plots(cohort1, cohort2, error_collumn, save_path, title):
    '''
    :param cohort1: list of string of all session paths with all cohort1 participants
    :param cohort2: list of string of all session paths with all cohort2 participants
    :param save_path: string of path where the plots will be saved to
    :param title: a title name for the comparison
    :return: plots in a new folder
    '''
    cohort1_results = get_compact_cohort_error_std_results(cohort1, error_collumn)
    cohort2_results = get_compact_cohort_error_std_results(cohort2, error_collumn)

    # now time to do the plots against eachother.


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #recording_folder_path = r"Z:\ActiveProjects\Harry\OculusVR\test_vr_recordings_jan20"
    recording_folder_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"

    # enter strings of path names for each participants (right into the session directory)
    cohort1=["",
             ""]
    cohort2=["",
             ""]
    save_path = ""

    plot_cohort_comparison_plots(cohort1, cohort2, error_collumn="first_stop_error", title="enter title name")

if __name__ == '__main__':
    main()



