import matplotlib.pyplot as plt
#from summarize.plot_behaviour_summary import *
from summarize.common import *
import numpy as np
# plotting functions, some taken from Sarah
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import curve_fit
from Modelling.no_kalman_model import *


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
        #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def plot_stops_in_time(trial_results, session_path):
    stops_in_time = plt.figure(figsize=(6, 6))
    ax = stops_in_time.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)
    # TODO

def plot_regression(ax, x, y):
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
    #ax.text(6, 0.65, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=2)))

    ax.text(  # position text relative to Axes
        0.95, 0.95, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=2)),
        ha='right', va='top',
        transform=ax.transAxes, fontsize=20)

    ax.plot(x_test, Y_pred, color='red')

def error_longer_tracks(trial_results, session_path, error_collumn):
    save_path = session_path+r'\Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    first_stop_errors = plt.figure(figsize=(6, 6))
    ax = first_stop_errors.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)

    uniques_lengths = np.unique(np.asarray(trial_results["integration_length"]))
    b_mean_errors_for_lengths = []
    b_std_errors_for_lengths = []
    nb_mean_errors_for_lengths = []
    nb_std_errors_for_lengths = []
    p_mean_errors_for_lengths = []
    p_std_errors_for_lengths = []

    for i in range(len(uniques_lengths)):
        b_errors = np.array(beaconed[error_collumn])[np.array(beaconed["integration_length"])==uniques_lengths[i]]
        if len(b_errors)<1:
            b_errors=np.nan
        b_mean_errors_for_lengths.append(np.nanmean(b_errors))
        b_std_errors_for_lengths.append(np.nanstd(b_errors))
        nb_errors = np.array(non_beaconed[error_collumn])[np.array(non_beaconed["integration_length"])==uniques_lengths[i]]
        if len(nb_errors)<1:
            nb_errors=np.nan
        nb_mean_errors_for_lengths.append(np.nanmean(nb_errors))
        nb_std_errors_for_lengths.append(np.nanstd(nb_errors))
        p_errors = np.array(probe[error_collumn])[np.array(probe["integration_length"])==uniques_lengths[i]]
        if len(p_errors)<1:
            p_errors=np.nan
        p_mean_errors_for_lengths.append(np.nanmean(p_errors))
        p_std_errors_for_lengths.append(np.nanstd(p_errors))

    ax.plot(uniques_lengths, b_mean_errors_for_lengths, color="k")
    ax.fill_between(uniques_lengths, np.array(b_mean_errors_for_lengths)-np.array(b_std_errors_for_lengths), np.array(b_mean_errors_for_lengths)+np.array(b_std_errors_for_lengths), facecolor="k", alpha=0.3)
    ax.plot(uniques_lengths, nb_mean_errors_for_lengths, color="r")
    ax.fill_between(uniques_lengths, np.array(nb_mean_errors_for_lengths)-np.array(nb_std_errors_for_lengths), np.array(nb_mean_errors_for_lengths)+np.array(nb_std_errors_for_lengths), facecolor="r", alpha=0.3)
    ax.plot(uniques_lengths, p_mean_errors_for_lengths, color="b")
    ax.fill_between(uniques_lengths, np.array(p_mean_errors_for_lengths)-np.array(p_std_errors_for_lengths), np.array(p_mean_errors_for_lengths)+np.array(p_std_errors_for_lengths), facecolor="b", alpha=0.3)

    for index, _ in beaconed.iterrows():
        error = np.array(beaconed[error_collumn][index])
        integration_length = np.array(beaconed["integration_length"][index])

        ax.scatter(integration_length, error, color="k", marker="x") # marks out track area
        #ax.plot(b_stops, b_trials, 'o', color='0.5', markersize=2)

    for index, _ in non_beaconed.iterrows():
        error = np.array(non_beaconed[error_collumn][index])
        integration_length = np.array(non_beaconed["integration_length"][index])

        ax.scatter(integration_length, error, color="r", marker="x") # marks out track area
        #ax.plot(nb_stops, nb_trials, 'o', color='red', markersize=2)

    for index, _ in probe.iterrows():
        error = np.array(probe[error_collumn][index])
        integration_length = np.array(probe["integration_length"][index])

        ax.scatter(integration_length, error, color="b", marker="x") # marks out track area
        #ax.plot(p_stops, p_trials, 'o', color='blue', markersize=2)

    plt.xlabel('Intergration Distance (vu)', fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.22, right=0.87, top=0.92)

    style_vr_plot(ax)  # can be any trialtype example
    if error_collumn == "absolute_first_stop_error":
        plt.ylabel('Absolute First Stop Error (vu)', fontsize=20, labelpad=10)
        plt.savefig(save_path + '/abs_error_plot.png', dpi=200)
    elif error_collumn == "first_stop_error":
        plt.ylabel('First Stop Error (vu)', fontsize=20, labelpad=10)
        plt.savefig(save_path + '/error_plot.png', dpi=200)
    elif error_collumn == "absolute_first_stop_post_cue_error":
        plt.ylabel('Absolute First Stop Post Cue Error (vu)', fontsize=20, labelpad=10)
        plt.savefig(save_path + '/abs_post_cue_error_plot.png', dpi=200)

    #plt.show()
    plt.close()

def concatenate_if_possible(x):
    if len(x) == 0:
        return x
    else:
        return np.hstack(x)

def stop_histogram(trial_results, session_path, cummulative=False, first_stop=False):

    bins = np.arange(-20, 340, 5)
    bin_centres = 0.5*(bins[1:]+bins[:-1])

    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)
    uniques_lengths = np.unique(np.asarray(trial_results["integration_length"]))

    stop_histogram1 = plt.figure(figsize=(6, 18))

    counter = 1
    for i in range(len(uniques_lengths)):
        beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)
        track_length_trial_results = trial_results[trial_results["integration_length"] ==uniques_lengths[i]]

        if first_stop:
            b_stop_locations = concatenate_if_possible(np.array(beaconed["first_stop_location_relative_to_ip"])[np.array(beaconed["integration_length"])==uniques_lengths[i]])
            nb_stop_locations = concatenate_if_possible(np.array(non_beaconed["first_stop_location_relative_to_ip"])[np.array(non_beaconed["integration_length"])==uniques_lengths[i]])
        else:
            b_stop_locations = concatenate_if_possible(np.array(beaconed["stop_locations_relative_to_ip"])[np.array(beaconed["integration_length"])==uniques_lengths[i]])
            nb_stop_locations = concatenate_if_possible(np.array(non_beaconed["stop_locations_relative_to_ip"])[np.array(non_beaconed["integration_length"])==uniques_lengths[i]])

        b_stop_hist = np.histogram(b_stop_locations, bins)[0]/np.sum(np.histogram(b_stop_locations, bins)[0])
        nb_stop_hist = np.histogram(nb_stop_locations, bins)[0]/np.sum(np.histogram(nb_stop_locations, bins)[0])

        if cummulative:
            b_stop_hist = np.cumsum(b_stop_hist)
            nb_stop_hist = np.cumsum(nb_stop_hist)

        ax = stop_histogram1.add_subplot(len(uniques_lengths), 1, counter)
        ax.plot(bin_centres, b_stop_hist, color="k")
        ax.plot(bin_centres, nb_stop_hist, color="r")

        plt.ylabel('Stop Probability', fontsize=20, labelpad=10)
        if counter == 5:
            plt.xlabel('Location (vu) relative to Integration Point', fontsize=20, labelpad=10)
        plt.xlim(-20, 340)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        style_vr_plot(ax)  # can be any trialtype example
        style_track_plot(ax, trial_results[trial_results["integration_length"] ==uniques_lengths[i]], one_length=True)
        bb_min = track_length_trial_results["Track End"].iloc[0]-track_length_trial_results["Cue Boundary Max"].iloc[0]
        bb_max = track_length_trial_results["Track End"].iloc[0]-track_length_trial_results["Cue Boundary Max"].iloc[0] + 20
        ax.axvspan(bb_min, bb_max, facecolor='black', alpha=.35, linewidth=0)
        counter +=1

    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.12, right=0.87, top=0.92)

    legend_elements = [Line2D([0], [0], marker="_", color='w', markeredgecolor="k", markerfacecolor='none', label='Beaconed'),
                       Line2D([0], [0], marker="_", color='w', markeredgecolor="r", markerfacecolor='none', label='Non Beaconed')]
    ax.legend(handles=legend_elements)

    if cummulative and first_stop:
        plt.savefig(session_path + '/first_stop_cumhistograms.png', dpi=200)
    elif cummulative and not first_stop:
        plt.savefig(session_path + '/stop_cumhistograms.png', dpi=200)
    elif not cummulative and first_stop:
        plt.savefig(session_path + '/first_stop_histograms.png', dpi=200)
    elif not cummulative and not first_stop:
        plt.savefig(session_path + '/stop_histograms.png', dpi=200)

    #plt.show()
    plt.close()

def variance_longer_tracks(trial_results, session_path, error_collumn):
    save_path = session_path+r'\Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    first_stop_errors = plt.figure(figsize=(6, 6))
    ax = first_stop_errors.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)
    uniques_lengths = np.unique(np.asarray(trial_results["integration_length"]))
    b_mean_errors_for_lengths = []
    b_std_errors_for_lengths = []
    nb_mean_errors_for_lengths = []
    nb_std_errors_for_lengths = []
    p_mean_errors_for_lengths = []
    p_std_errors_for_lengths = []

    for i in range(len(uniques_lengths)):
        b_errors = np.array(beaconed[error_collumn])[np.array(beaconed["integration_length"])==uniques_lengths[i]]
        if len(b_errors)<1:
            b_errors=np.nan
        b_mean_errors_for_lengths.append(np.nanmean(b_errors))
        b_std_errors_for_lengths.append(np.nanstd(b_errors))

        nb_errors = np.array(non_beaconed[error_collumn])[np.array(non_beaconed["integration_length"])==uniques_lengths[i]]
        if len(nb_errors)<1:
            nb_errors=np.nan
        nb_mean_errors_for_lengths.append(np.nanmean(nb_errors))
        nb_std_errors_for_lengths.append(np.nanstd(nb_errors))

        p_errors = np.array(probe[error_collumn])[np.array(probe["integration_length"])==uniques_lengths[i]]
        if len(p_errors)<1:
            p_errors=np.nan
        p_mean_errors_for_lengths.append(np.nanmean(p_errors))
        p_std_errors_for_lengths.append(np.nanstd(p_errors))

    ax.plot(uniques_lengths, b_std_errors_for_lengths, color="k")
    ax.plot(uniques_lengths, nb_std_errors_for_lengths, color="r")
    ax.plot(uniques_lengths, p_std_errors_for_lengths, color="b")

    plt.xlabel('Intergration Distance (vu)', fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.22, right=0.87, top=0.92)

    style_vr_plot(ax)  # can be any trialtype example
    if error_collumn == "absolute_first_stop_error":
        plt.ylabel('Variance of Absolute First Stop Error', fontsize=20, labelpad=10)
        plt.savefig(save_path + '/variance_abs_error_plot.png', dpi=200)
    elif error_collumn == "first_stop_error":
        plt.ylabel('Variance of First Stop Error', fontsize=20, labelpad=10)
        plt.savefig(save_path + '/Variance_error_plot.png', dpi=200)
    elif error_collumn =="absolute_first_stop_post_cue_error":
        plt.ylabel('Variance of First Stop Post Cue Error', fontsize=20, labelpad=10)
        plt.savefig(save_path + '/Variance_abs_post_cue_error_plot.png', dpi=200)

    #plt.show()
    plt.close()

def plot_stops_on_track(trial_results, session_path):
    save_path = session_path+r'\Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    stops_on_track = plt.figure(figsize=(6, 6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)

    for index, _ in beaconed.iterrows():
        b_stops = np.array(beaconed["stop_locations"][index])-beaconed["Cue Boundary Max"][index]
        b_trial_num = np.array(beaconed["trial_num_in_block"][index])
        b_trials = b_trial_num*np.ones(len(b_stops))

        ax.plot((np.linspace(beaconed["Track Start"][index], beaconed["Track End"][index], 2))-beaconed["Cue Boundary Max"][index], np.array([b_trial_num, b_trial_num]), color="y") # marks out track area
        ax.plot(b_stops, b_trials, 'o', color='black', markersize=2)

    for index, _ in non_beaconed.iterrows():
        nb_stops = (np.array(non_beaconed["stop_locations"][index]))-non_beaconed["Cue Boundary Max"][index]
        nb_trial_num = np.array(non_beaconed["trial_num_in_block"][index])
        nb_trials = nb_trial_num * np.ones(len(nb_stops))

        ax.plot((np.linspace(non_beaconed["Track Start"][index], non_beaconed["Track End"][index], 2))-non_beaconed["Cue Boundary Max"][index], np.array([nb_trial_num,nb_trial_num]), color="y")  # marks out track area
        ax.plot(nb_stops, nb_trials, 'o', color='red', markersize=2)

    for index, _ in probe.iterrows():
        p_stops = (np.array(probe["stop_locations"][index]))-probe["Cue Boundary Max"][index]
        p_trial_num = np.array(probe["trial_num_in_block"][index])
        p_trials = p_trial_num * np.ones(len(p_stops))

        ax.plot((np.linspace(probe["Track Start"][index], probe["Track End"][index], 2))-probe["Cue Boundary Max"][index], np.array([p_trial_num, p_trial_num]), color="y")  # marks out track area
        ax.plot(p_stops, p_trials, 'o', color='blue', markersize=2)

    plt.ylabel('Trial Number', fontsize=20, labelpad=10)
    plt.xlabel('Location (vu) relative to Integration Point', fontsize=20, labelpad=10)
    # plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    #plt.xlim(0, 200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    style_track_plot(ax, trial_results)
    #style_track_plot(ax, beaconed)
    #style_track_plot(ax, non_beaconed)

    style_vr_plot(ax)  # can be any trialtype example

    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.12, right=0.87, top=0.92)
    plt.savefig(save_path + '/summary_plot.png', dpi=200)
    #plt.savefig('/home/harry/aa/plot_summary.png', dpi=200)   # TODO change this to ardbeg when I have permission to write with Linux
    #plt.show()
    plt.close()


def plot_target_response(trial_results, session_path):
    save_path = session_path+r'\Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    trial_results = trial_results.dropna()
    trial_results_means = trial_results.groupby(['Trial type', 'integration_length', 'ppid'])['first_stop_location'].mean().reset_index()
    ppid = trial_results_means["ppid"][0]

    subject_data_mean_x = np.asarray(trial_results_means[(trial_results_means["Trial type"] == "non_beaconed")]['integration_length'])
    subject_data_mean_y = np.asarray(trial_results_means[(trial_results_means["Trial type"] == "non_beaconed")]['first_stop_location'])

    subject_data_x = np.asarray(trial_results[(trial_results["Trial type"] == "non_beaconed")]['integration_length'])
    subject_data_y = np.asarray(trial_results[(trial_results["Trial type"] == "non_beaconed")]["first_stop_location"])

    _popt = []
    minimal = 1e16
    minimal_i = 0
    test_x = np.arange(0,400, 20)

    for i in range(100):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=0, high=2) # gamma
        p0_2 = np.random.uniform(low=0, high=2) # lambda
        p0_3 = np.random.uniform(low=0, high=0.1) # k

        popt, pcov = curve_fit(model, subject_data_mean_x, subject_data_mean_y, p0=[p0_1, p0_2, p0_3])
        sq_sum_error = np.sum(np.square(model(subject_data_mean_x, prior_gain=popt[0], lambda_coef=popt[1], k=popt[2]) - subject_data_mean_y))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)


    subject_model_params = _popt[minimal_i]
    print("estimate of model parameters ", subject_model_params)
    # plotting model fit
    best_fit_responses = model(test_x, prior_gain=subject_model_params[0], lambda_coef=subject_model_params[1], k=subject_model_params[2])
    # plot optimised response target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.title(ppid, fontsize="20")
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
        r'$\Gamma=%.2f$' % (_popt[minimal_i][0], ),
        r'$\lambda=%.2f$' % (_popt[minimal_i][1], ),
        r'$\mathrm{k}=%.2f$' % (_popt[minimal_i][2], )))

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)

    plt.legend(loc="upper left")
    plt.savefig(save_path+"/"+ppid+"_model_fit.png")
    plt.show()
    plt.close()

def plot_speed_on_track(trial_results, session_path, block=2):
    # TODO
    pass


def style_vr_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True,
        labelsize=14,
        length=5,
        width=1.5)  # labels along the bottom edge are off

    #ax.set_aspect('equal')
    return ax

def style_track_plot(ax, dataframe, one_length=False):
    '''
    plots reward and cue locations
    :param ax:
    :param dataframe: can be any dataframe that has collumns with reward zone bounds and allows cue bounds
    :return:
    '''

    if one_length:
        ax.axvspan(dataframe["Reward Boundary Min"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0],
                   dataframe["Reward Boundary Max"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0], facecolor='DarkGreen', alpha=.25, linewidth=0)


    else:
        reward_zone_min = np.asarray(dataframe["Reward Boundary Min"]-dataframe["Cue Boundary Max"])
        reward_zone_max = np.asarray(dataframe["Reward Boundary Max"]-dataframe["Cue Boundary Max"])
        trial_numbers = np.asarray(dataframe["trial_num_in_block"])
        #ax.axvspan(reward_zone_min, reward_zone_max, facecolor='DarkGreen', alpha=.25, linewidth=0)

        for i in range(len(reward_zone_max)):
            x = [reward_zone_min[i],
                 reward_zone_max[i],
                 reward_zone_max[i],
                 reward_zone_min[i]]
            y = [trial_numbers[i]-0.5, trial_numbers[i]-0.5, trial_numbers[i]+0.5, trial_numbers[i]+0.5]
            ax.fill(x, y, alpha=0.25, color="g")

    if dataframe["Cue Boundary Min"].iloc[0] > 0:
        cue_min = dataframe["Cue Boundary Min"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0]
        cue_max = dataframe["Cue Boundary Max"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0]

        if np.sum(np.asarray(dataframe["Transparency"]))>0: # only plot cue zone if cue Transparency is greater than 0
            ax.axvspan(cue_min, cue_max, facecolor='plum', alpha=.25, linewidth=0)