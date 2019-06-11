import matplotlib.pyplot as plt
# plotting functions, some taken from Sarah

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
        cue_min = dataframe["Cue Boundary Min"].iloc[0]*-1+dataframe["Cue Boundary Max"].iloc[0]
        cue_max = dataframe["Cue Boundary Max"].iloc[0]*-1+dataframe["Cue Boundary Max"].iloc[0]
        ax.axvspan(cue_min, cue_max, facecolor='plum', alpha=.25, linewidth=0)

    reward_zone_min = dataframe["Reward Boundary Min"].iloc[0]*-1+dataframe["Cue Boundary Max"].iloc[0]
    reward_zone_max = dataframe["Reward Boundary Max"].iloc[0]*-1+dataframe["Cue Boundary Max"].iloc[0]
    ax.axvspan(reward_zone_min, reward_zone_max, facecolor='DarkGreen', alpha=.25, linewidth=0)

    #ax.axvspan(x1, x2, facecolor='DarkGreen', alpha=.25, linewidth =0)
    #ax.axvspan(0, 30/divider, facecolor='k', linewidth =0, alpha=.25) # black box
    #ax.axvspan((200-30)/divider, 200/divider, facecolor='k', linewidth =0, alpha=.25)# black box