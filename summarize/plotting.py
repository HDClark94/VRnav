import matplotlib.pyplot as plt
#from summarize.plot_behaviour_summary import *
from summarize.common import *
# plotting functions, some taken from Sarah

def plot_stops_in_time(trial_results, session_path):
    stops_in_time = plt.figure(figsize=(6, 6))
    ax = stops_in_time.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)
    # TODO


def plot_stops_on_track(trial_results, session_path):
    stops_on_track = plt.figure(figsize=(6, 6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)

    for index, _ in beaconed.iterrows():
        b_stops = np.array(beaconed["stop_locations"][index])-beaconed["Cue Boundary Max"][index]
        b_trial_num = np.array(beaconed["trial_num"][index])
        b_trials = b_trial_num*np.ones(len(b_stops))

        ax.plot((np.linspace(beaconed["Track Start"][index], beaconed["Track End"][index], 2))-beaconed["Cue Boundary Max"][index], np.array([b_trial_num, b_trial_num]), color="y") # marks out track area
        ax.plot(b_stops, b_trials, 'o', color='0.5', markersize=2)

    for index, _ in non_beaconed.iterrows():
        nb_stops = (np.array(non_beaconed["stop_locations"][index]))-non_beaconed["Cue Boundary Max"][index]
        nb_trial_num = np.array(non_beaconed["trial_num"][index])
        nb_trials = nb_trial_num * np.ones(len(nb_stops))

        ax.plot((np.linspace(non_beaconed["Track Start"][index], non_beaconed["Track End"][index], 2))-non_beaconed["Cue Boundary Max"][index], np.array([nb_trial_num,nb_trial_num]), color="y")  # marks out track area
        ax.plot(nb_stops, nb_trials, 'o', color='red', markersize=2)

    for index, _ in probe.iterrows():
        p_stops = (np.array(probe["stop_locations"][index]))-probe["Cue Boundary Max"][index]
        p_trial_num = np.array(probe["trial_num"][index])
        p_trials = p_trial_num * np.ones(len(p_stops))

        ax.plot((np.linspace(probe["Track Start"][index], probe["Track End"][index], 2))-probe["Cue Boundary Max"][index], np.array([p_trial_num, p_trial_num]), color="y")  # marks out track area
        ax.plot(p_stops, p_trials, 'o', color='blue', markersize=2)

    plt.ylabel('Stops on trials', fontsize=12, labelpad=10)
    plt.xlabel('Location (vu)', fontsize=12, labelpad=10)
    # plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    #plt.xlim(0, 200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    style_track_plot(ax, beaconed)
    style_vr_plot(ax)  # can be any trialtype example

    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.12, right=0.87, top=0.92)
    plt.savefig(session_path + '/summary_plot.png', dpi=200)
    #plt.savefig('/home/harry/aa/plot_summary.png', dpi=200)   # TODO change this to ardbeg when I have permission to write with Linux
    #plt.show()
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

def style_track_plot(ax, dataframe):
    '''
    plots reward and cue locations
    :param ax:
    :param dataframe: can be any dataframe that has collumns with reward zone bounds and allows cue bounds
    :return:
    '''

    if dataframe["Cue Boundary Min"].iloc[0] is not 0:
        cue_min = dataframe["Cue Boundary Min"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0]
        cue_max = dataframe["Cue Boundary Max"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0]

        if np.sum(np.asarray(dataframe["Transparency"]))>0: # only plot cue zone if cue Transparency is greater than 0
            ax.axvspan(cue_min, cue_max, facecolor='plum', alpha=.25, linewidth=0)

    reward_zone_min = dataframe["Reward Boundary Min"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0]
    reward_zone_max = dataframe["Reward Boundary Max"].iloc[0]-dataframe["Cue Boundary Max"].iloc[0]
    ax.axvspan(reward_zone_min, reward_zone_max, facecolor='DarkGreen', alpha=.25, linewidth=0)

    #ax.axvspan(x1, x2, facecolor='DarkGreen', alpha=.25, linewidth =0)
    #ax.axvspan(0, 30/divider, facecolor='k', linewidth =0, alpha=.25) # black box
    #ax.axvspan((200-30)/divider, 200/divider, facecolor='k', linewidth =0, alpha=.25)# black box