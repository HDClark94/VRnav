# -*- coding: utf-8 -*-
"""

Two part script:
### 1. Calculates average discrimination indexes for both telc and gfp mice
### 2. plots for both object recognition and object location


"""


# import packages and functions
from Functions_CoreFunctions_0100 import adjust_spines, makelegend2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

# Load data containing discrimination indexes
marray = np.loadtxt('Data_Input/Behaviour_SummaryData/ExplorationTasks_0100.txt',delimiter = '\t')

# specify mouse/mice  to analyse
mice = ['M' + str(int(x)) for x in np.arange(1,16.1)]# choose specific day/s


#FUNCTION FOR PLOTTING LEGEND
def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="right", bbox_to_anchor=(0.85, 0.58), fontsize = "small", markerscale = 1)
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)


# Get scores for each mouse
tetarray_loc = np.zeros((len(mice))); tetarray_loc[:] = np.nan
conarray_loc = np.zeros((len(mice))); conarray_loc[:] = np.nan
tetarray_ob = np.zeros((len(mice))); tetarray_ob[:] = np.nan
conarray_ob = np.zeros((len(mice))); conarray_ob[:] = np.nan

for mcount,mouse in enumerate(mice):
    objectscore = marray[mcount,1]
    locationscore = marray[mcount,0]
    if marray[mcount,2] == 2:
        tetarray_ob[mcount] = objectscore
        tetarray_loc[mcount] = locationscore
    if marray[mcount,2] == 1:
        conarray_ob[mcount] = objectscore
        conarray_loc[mcount] = locationscore



tetarray_ob = tetarray_ob[~np.isnan(tetarray_ob)]
tetarray_loc = tetarray_loc[~np.isnan(tetarray_loc)]
conarray_ob = conarray_ob[~np.isnan(conarray_ob)]
conarray_loc = conarray_loc[~np.isnan(conarray_loc)]


objectscores_tet = np.nanmean(tetarray_ob[:])
locationscores_tet = np.nanmean(tetarray_loc[:])
sdobjectscores_tet = np.nanstd(tetarray_ob[:])/np.sqrt(8)
sdlocationscores_tet = np.nanstd(tetarray_loc[:])/np.sqrt(8)

objectscores_con = np.nanmean(conarray_ob[:])
locationscores_con = np.nanmean(conarray_loc[:])
sdobjectscores_con = np.nanstd(conarray_ob[:])/np.sqrt(8)
sdlocationscores_con = np.nanstd(conarray_loc[:])/np.sqrt(8)



objectrec = [objectscores_con,objectscores_tet]
sdobjectrec = [sdobjectscores_con,sdobjectscores_tet]

locationrec = [locationscores_con,locationscores_tet]
sdlocationrec = [sdlocationscores_con,sdlocationscores_tet]


# plot graphs

index1 = np.hstack((1, 2))
n_groups = np.arange(3)
bar_width = 0.5


fig = plt.figure(figsize = (4,5.5))
ax = fig.add_subplot(111)

ax.plot(1,objectscores_con, 'o', color = 'k')
ax.errorbar(1,objectscores_con,sdobjectscores_con, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,objectscores_tet, 'o', color = 'k')
ax.errorbar(2,objectscores_tet,sdobjectscores_tet, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)

ax.plot(np.hstack((1,1,1,1,1,1,1,1)),conarray_ob, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2,2,2)),tetarray_ob, 'o', color = 'blue', alpha = 0.5, markersize = 10)


adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(-0.1,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
ax.set_ylim(-0.1,0.81)
ax.set_xlim(0.5,2.5)
plt.locator_params(axis = 'y', nbins  = 6)
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)
fig.savefig('Plots/Figure6/ObjectRec_0100' +' .png', dpi = 200)
plt.close()


fig = plt.figure(figsize = (4,5.5))
ax = fig.add_subplot(111)
ax.plot(1,locationscores_con, 'o', color = 'k')
ax.errorbar(1,locationscores_con,sdlocationscores_con, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,locationscores_tet, 'o', color = 'k')
ax.errorbar(2,locationscores_tet,sdlocationscores_tet, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)

ax.plot(np.hstack((1,1,1,1,1,1,1,1)),conarray_loc, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2,2,2)),tetarray_loc, 'o', color = 'blue', alpha = 0.5, markersize = 10)

adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(-0.1,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
ax.set_ylim(-0.1,0.81)
ax.set_xlim(0.5,2.5)
plt.locator_params(axis = 'y', nbins  = 6)
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)

plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)
fig.savefig('Plots/Figure6/ObjectLoc_0100' +' .png', dpi = 200)
plt.close()

