import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from Modelling.no_kalman_model import *
from Modelling.Stochastic_angelaki import *
from scipy import stats
from scipy.optimize import least_squares
from Modelling.params import Parameters
import matplotlib.ticker as ticker

# we set the seed so results that rely on random number generation can be consistently reproduced.
np.random.seed(64)

'''
this script was written to model fit target response data to the Angelaki dynamic baseysian observer model
add reference (2018)

this script takes processed_results pandas dataframes created in concantenate_data.py
see /../summarize/

some functions within this script includes early attempts to fit stochastic model, this is still mostly wrong
'''

def get_std_with_id(human_data, human_std_data, y_column):
    y_std = []
    tmp_columns = human_std_data.columns.to_list()
    tmp_columns.remove(y_column)

    for index, row in human_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)

        tt = row["Trial type"].iloc[0]
        il = row["integration_length"].iloc[0]
        ppid = row["ppid"].iloc[0]

        corresponding_row = human_std_data[((human_std_data["Trial type"] == tt) &
                                            (human_std_data["integration_length"] == il) &
                                            (human_std_data["ppid"] == ppid))]

        std = corresponding_row[y_column].iloc[0]

        y_std.append(std)
    human_data["y_std"] = y_std
    return human_data

def get_std_with_group(human_data, human_std_data, y_column):
    y_std = []
    tmp_columns = human_std_data.columns.to_list()
    tmp_columns.remove(y_column)

    for index, row in human_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)

        tt = row["Trial type"].iloc[0]
        il = row["integration_length"].iloc[0]

        corresponding_row = human_std_data[((human_std_data["Trial type"] == tt) &
                                            (human_std_data["integration_length"] == il))]

        std = corresponding_row[y_column].iloc[0]
        y_std.append(std)

    human_data["y_std"] = y_std
    return human_data



def fit_parameters_to_model(x, y, sigmas, starters=[(0,2),(0,2),(0,0.1)]):
    _popt = []
    minimal = 1e16
    minimal_i = 0

    for i in range(100):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=starters[0][0], high=starters[0][1])
        p0_2 = np.random.uniform(low=starters[1][0], high=starters[1][1])
        p0_3 = np.random.uniform(low=starters[2][0], high=starters[2][1])

        popt, pcov = curve_fit(model, x, y, sigma=1/sigmas, p0=[p0_1, p0_2, p0_3])
        sq_sum_error = np.sum(np.square(model(x, prior_gain=popt[0], lambda_coef=popt[1], k=popt[2]) - y))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)

    model_params = _popt[minimal_i]
    print("estimate of model parameters ", model_params)
    return model_params

def fit_parameters_to_Stochastic_model(x, y, starters=[(0,1),(0,1),(0,1),(0,1)]):
    params = Parameters(gain=1, dt=1, n_trials=10, N_iterations=500, velocity_max = 2, target_width=0.1)

    _popt = []
    minimal = 1e16
    minimal_i = 0

    for i in range(50):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=starters[0][0], high=starters[0][1])
        p0_2 = np.random.uniform(low=starters[1][0], high=starters[1][1])
        p0_3 = np.random.uniform(low=starters[2][0], high=starters[2][1])
        p0_4 = np.random.uniform(low=starters[3][0], high=starters[3][1])

        popt, pcov = curve_fit(model_stochastic, x, y, p0=[p0_1, p0_2, p0_3, p0_4])
        sq_sum_error = np.sum(np.square(model_stochastic(x, likelihood_width=popt[0], expon_coef=popt[1], lambda_coef=popt[2], k=popt[3]) - y))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)

    model_params = _popt[minimal_i]
    print("estimate of model parameters ", model_params)
    return model_params

def fit(human_data_original, save_path, trial_type, model_parameters_df=None, constraints=""):
    '''
    :param human_data: processed results pandas dataframe
    :param save_path: the path to save dataframes and figures to
    :param trial_type: the trial type to model fit
    :param constraints: collumn title of processed_results for which you would like to fit seperately for each condition of
    the given constraint
    :return: model fit plots
            model fit parameter dataframe
    '''
    test_x = np.arange(0,400, 20)

    if model_parameters_df is None:
        model_parameters_df = pd.DataFrame()

    collumn_values = [""]
    if constraints is not "":
        collumn_values = np.unique(human_data_original[constraints])

        for collumn_value in collumn_values:

            # only filter results by the constraint if a constraint is parsed
            if collumn_value is not "":
                human_data = human_data_original[(human_data_original[constraints] == collumn_value)]
            else:
                human_data = human_data_original

            human_data = human_data.dropna()
            human_mean_data = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].mean().reset_index()
            human_mean_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()
            human_std_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].std().reset_index()
            human_std_data_group = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].std().reset_index()

            human_mean_with_id_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['integration_length'])
            human_mean_with_id_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['first_stop_location'])
            human_mean_data_x = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['integration_length'])
            human_mean_data_y = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['first_stop_location'])
            human_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['integration_length'])
            human_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['first_stop_location'])

            human_data_with_std_individual = get_std_with_id(human_data, human_std_data_with_id, y_column='first_stop_location')
            human_data_with_std_group = get_std_with_group(human_data, human_std_data_group,  y_column='first_stop_location')
            human_data_std_id = np.asarray(human_data_with_std_individual[(human_data_with_std_individual["Trial type"] == trial_type)]['y_std'])
            human_data_std_group = np.asarray(human_data_with_std_group[(human_data_with_std_group["Trial type"] == trial_type)]['y_std'])

            group_model_params = fit_parameters_to_model(human_data_x, human_data_y, human_data_std_group)

            # plotting model fit
            best_fit_responses = model(test_x, prior_gain=group_model_params[0], lambda_coef=group_model_params[1], k=group_model_params[2])
            # plot optimised response target
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(1,1,1) #stops per trial
            plt.title("All subjects", fontsize="20")
            plt.scatter(human_mean_with_id_x, human_mean_with_id_y, color="r", marker="o")
            plt.plot(human_mean_data_x, human_mean_data_y, "r", label="data")
            plt.plot(test_x, best_fit_responses, "g", label="model")
            plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
            plt.xlabel("Target (VU)", fontsize=20)
            plt.xlim((0,400))
            plt.ylim((0,400))
            plt.ylabel("Response (VU)", fontsize=20)
            plt.subplots_adjust(left=0.2)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            textstr = '\n'.join((
                r'$\Gamma=%.2f$' % (group_model_params[0], ),
                r'$\lambda=%.2f$' % (group_model_params[1], ),
                r'$\mathrm{k}=%.2f$' % (group_model_params[2], )))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
            plt.legend(loc="upper left")

            if constraints is not "":
                plt.savefig(save_path+"/"+trial_type+"_"+constraints+remove_dots(str(collumn_value))+"_group_model_fit.png")
            else:
                plt.savefig(save_path+"/"+trial_type+"_"+constraints+"_group_model_fit.png")

            plt.show()
            plt.close()

            # save the group model parameters
            group_model_parameters_df = pd.DataFrame()
            group_model_parameters_df["ppid"] = ["group"]
            group_model_parameters_df["gamma"] = [group_model_params [0]]
            group_model_parameters_df["lambda"] = [group_model_params [1]]
            group_model_parameters_df["k"] = [group_model_params [2]]
            group_model_parameters_df["trial_type"] = [trial_type]
            if constraints is not "":
                group_model_parameters_df[constraints] = [collumn_value] # adds the constraint condition
            model_parameters_df = pd.concat([model_parameters_df, group_model_parameters_df], ignore_index=True)


            ppids = np.unique(human_data["ppid"])
            subjects_model_params = np.zeros((len(ppids), 3)) # fitting 3 parameters

            for j in range(len(ppids)):
                subject_model_parameters_df = pd.DataFrame()

                subject_data_mean_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['integration_length'])
                subject_data_mean_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['first_stop_location'])

                subject_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]['integration_length'])
                subject_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]["first_stop_location"])

                subject_human_data_with_std_individual = np.asarray(human_data_with_std_individual[(human_data_with_std_individual["Trial type"] == trial_type) &
                                                                                                   (human_data_with_std_individual["ppid"] == ppids[j])]["y_std"])

                subject_model_params = fit_parameters_to_model(subject_data_x, subject_data_y, subject_human_data_with_std_individual)
                subjects_model_params[j] = subject_model_params

                subject_model_parameters_df["ppid"] = [ppids[j]]
                subject_model_parameters_df["gamma"] = [subject_model_params [0]]
                subject_model_parameters_df["lambda"] = [subject_model_params [1]]
                subject_model_parameters_df["k"] = [subject_model_params [2]]
                subject_model_parameters_df["trial_type"] = [trial_type]
                if constraints is not "":
                    subject_model_parameters_df[constraints] = [collumn_value] # adds the constraint condition
                model_parameters_df = pd.concat([model_parameters_df, subject_model_parameters_df], ignore_index=True)

                # plotting model fit
                best_fit_responses = model(test_x, prior_gain=subject_model_params[0], lambda_coef=subject_model_params[1], k=subject_model_params[2])
                # plot optimised response target
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(1,1,1) #stops per trial
                plt.title(ppids[j], fontsize="20")
                plt.scatter(subject_data_x, subject_data_y, color="r", marker="o")
                plt.plot(subject_data_mean_x, subject_data_mean_y, "r", label="data")
                plt.plot(test_x, best_fit_responses, "g", label="model")
                plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
                plt.xlabel("Target", fontsize=20)
                plt.xlim((0,400))
                plt.ylim((0,400))
                plt.ylabel("Optimal Response", fontsize=20)
                plt.subplots_adjust(left=0.2)
                ax.tick_params(axis='both', which='major', labelsize=15)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                textstr = '\n'.join((
                    r'$\Gamma=%.2f$' % (subject_model_params[0], ),
                    r'$\lambda=%.2f$' % (subject_model_params[1], ),
                    r'$\mathrm{k}=%.2f$' % (subject_model_params[2], )))
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
                plt.legend(loc="upper left")
                plt.savefig(save_path+"/"+trial_type+"_"+ppids[j]+"_"+constraints+remove_dots(str(collumn_value))+"_model_fit.png")
                plt.show()
                plt.close()

    return model_parameters_df

def remove_participants(data, participants_to_remove):
    for i in range(len(participants_to_remove)):
        data = data[(data["ppid"] != participants_to_remove[i])]
    return data

def remove_dots(value_string):
    tmp = value_string.split(".")
    return "".join(tmp)

def get_color_from_trial_type(trial_type):
    if trial_type == "beaconed":
        return "black"
    if trial_type == "non_beaconed":
        return "red"

def compare_constraints(human_data_original, save_path, trial_type, constraint):
    '''
    :param human_data: processed results pandas dataframe
    :param save_path: the path to save dataframes and figures to
    :param trial_type: the trial type to model fit
    :param constraints: collumn title of processed_results for which you would like to fit seperately for each condition of
    the given constraint
    '''
    collumn_values = np.unique(human_data_original[constraint])

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.title("All subjects", fontsize="20")
    plt.plot(np.arange(0,400), np.arange(0,400), "k--")

    for collumn_value, ls, alpha, shift, c in zip(collumn_values, ["solid", "solid"], [0.3, 0.3], [-3, 3], ["blue", "orangered"]):
        human_data = human_data_original[(human_data_original[constraint] == collumn_value)]
        human_data = human_data.dropna()

        # data average over people
        human_mean_grouped = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].mean().reset_index()
        human_sem_grouped = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].sem().reset_index()
        human_mean_data_x = np.asarray(human_mean_grouped[(human_mean_grouped["Trial type"] == trial_type)]['integration_length'])
        human_mean_data_y = np.asarray(human_mean_grouped[(human_mean_grouped["Trial type"] == trial_type)]['first_stop_location'])
        human_sem_data_y  = np.asarray(human_sem_grouped[(human_sem_grouped["Trial type"] == trial_type)]['first_stop_location'])

        # data not averaged over people
        human_mean_by_individual = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()
        human_sem_by_individual = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].sem().reset_index()
        human_mean_by_individual_x = np.asarray(human_mean_by_individual[(human_mean_by_individual["Trial type"] == trial_type)]['integration_length'])
        human_mean_by_individual_y = np.asarray(human_mean_by_individual[(human_mean_by_individual["Trial type"] == trial_type)]['first_stop_location'])

        # plot optimised response target
        plt.scatter(human_mean_by_individual_x+shift, human_mean_by_individual_y, color=c, marker="o")

        plt.plot(human_mean_data_x, human_mean_data_y, linestyle=ls, color=c)
        plt.fill_between(human_mean_data_x, human_mean_data_y-human_sem_data_y,
                         human_mean_data_y+human_sem_data_y, facecolor=c, alpha=alpha)

    plt.xlabel("Target (VU)", fontsize=20)
    plt.xlim((0,400))
    plt.ylim((0,400))
    tick_spacing = 100
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.ylabel("Response (VU)", fontsize=20)
    plt.subplots_adjust(left=0.2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(save_path+"\\"+trial_type+"_"+constraint+"_comparison.png", dpi=200)
    plt.show()
    plt.close()


    ppids = np.unique(human_data["ppid"])
    for j in range(len(ppids)):
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1) #stops per trial
        plt.title(ppids[j], fontsize="20")
        plt.plot(np.arange(0,400), np.arange(0,400), "k--")

        for collumn_value, ls, alpha, shift, c in zip(collumn_values, ["solid", "solid"], [0.3, 0.3], [-3, 3], ["blue", "orangered"]):
            human_data = human_data_original[(human_data_original[constraint] == collumn_value)]
            human_data = human_data.dropna()
            human_data = human_data[(human_data["ppid"] == ppids[j])]

            # data not averaged over people
            human_mean_by_individual = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()
            human_sem_by_individual = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].sem().reset_index()

            subject_data_mean_x = np.asarray(human_mean_by_individual[((human_mean_by_individual["Trial type"] == trial_type) & (human_mean_by_individual["ppid"] == ppids[j]))]['integration_length'])
            subject_data_mean_y = np.asarray(human_mean_by_individual[(human_mean_by_individual["Trial type"] == trial_type) & (human_mean_by_individual["ppid"] == ppids[j])]['first_stop_location'])
            subject_data_sem_y = np.asarray(human_sem_by_individual[(human_sem_by_individual["Trial type"] == trial_type) & (human_sem_by_individual["ppid"] == ppids[j])]['first_stop_location'])

            subject_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]['integration_length'])
            subject_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]["first_stop_location"])

            # plot optimised response target
            plt.scatter(subject_data_x+shift, subject_data_y-shift, color=c, marker="o")
            plt.plot(subject_data_mean_x, subject_data_mean_y, color=c)
            plt.fill_between(subject_data_mean_x, subject_data_mean_y-subject_data_sem_y,
                             subject_data_mean_y+subject_data_sem_y, facecolor=c, alpha=0.3)

        plt.xlabel("Target (VU)", fontsize=20)
        plt.xlim((0,400))
        plt.ylim((0,400))
        plt.ylabel("Response (VU)", fontsize=20)
        plt.subplots_adjust(left=0.2)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(save_path+"\\"+trial_type+"_"+ppids[j]+"_"+constraint+"_comparison.png")
        plt.show()
        plt.close()

def main():
    print("run something here")
    '''
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model"
    model_params = pd.DataFrame()
    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\processed_results.pkl")
    model_params = fit(human_data, save_path, trial_type="beaconed", model_parameters_df=model_params, constraints="gain_std")
    model_params = fit(human_data, save_path, trial_type="non_beaconed", model_parameters_df=model_params, constraints="gain_std")
    model_params.to_pickle(save_path+"/model_parameters.pkl")
    
    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Maya\processed_results.pkl")
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model"
    model_params = fit(human_data, save_path, trial_type="beaconed", model_parameters_df=model_params, constraints="movement_mechanism")
    model_params = fit(human_data, save_path, trial_type="non_beaconed", model_parameters_df=model_params, constraints="movement_mechanism")
    '''

    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\audio_visual"
    model_params = pd.DataFrame()
    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV4.0\recordings\processed_results.pkl")
    human_data = remove_participants(human_data, participants_to_remove=["P_210302132218"])
    compare_constraints(human_data, save_path=r"Z:\ActiveProjects\Harry\OculusVR\Figures\audio_visual\constraint_comparison", trial_type="beaconed", constraint="experiment")
    compare_constraints(human_data, save_path=r"Z:\ActiveProjects\Harry\OculusVR\Figures\audio_visual\constraint_comparison", trial_type="non_beaconed", constraint="experiment")

    model_params = fit(human_data, save_path=save_path, trial_type="beaconed", model_parameters_df=model_params, constraints="experiment")
    model_params = fit(human_data, save_path=save_path, trial_type="non_beaconed", model_parameters_df=model_params, constraints="experiment")
    model_params.to_pickle(save_path+"/model_parameters.pkl")

    human_data = pd.read_csv(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV4.0\recordings\processed_results.csv")
    human_data.to_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV4.0\recordings\processed_results.pkl")


if __name__ == '__main__':
    main()