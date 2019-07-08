# -*- coding: utf-8 -*-
"""

Two part script:
### 1. Calculates average object exploration for both telc and gfp mice
### 2. plots for both object recognition and object location


"""


# import packages and functions
from Functions_CoreFunctions_0100 import adjust_spines, makelegend2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

# Load data
marray = np.loadtxt('Data_Input/Behaviour_SummaryData/TotalExploration_ObjectLocation.txt',delimiter = '\t')
narray = np.loadtxt('Data_Input/Behaviour_SummaryData/TotalExploration_ObjectRecognition.txt',delimiter = '\t')

# specify mouse/mice and day/s to analyse
mice = ['M' + str(int(x)) for x in np.arange(1,16.1)]# choose specific day/s

#plot legend
def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="right", bbox_to_anchor=(0.85, 0.58), fontsize = "small", markerscale = 1)
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

# split data according to phase
obloc_sample = marray [0:16,:]
obloc_test = marray [16:32,:]
obrec_sample = narray [0:16,:]
obrec_test = narray [16:32,:]

# arrays for storing data
tetarray_loc_test = np.zeros((len(mice))); tetarray_loc_test[:] = np.nan
conarray_loc_test = np.zeros((len(mice))); conarray_loc_test[:] = np.nan
tetarray_loc_sample = np.zeros((len(mice))); tetarray_loc_sample[:] = np.nan
conarray_loc_sample = np.zeros((len(mice))); conarray_loc_sample[:] = np.nan
tetarray_ob_test = np.zeros((len(mice))); tetarray_ob_test[:] = np.nan
conarray_ob_test = np.zeros((len(mice))); conarray_ob_test[:] = np.nan
tetarray_ob_sample = np.zeros((len(mice))); tetarray_ob_sample[:] = np.nan
conarray_ob_sample = np.zeros((len(mice))); conarray_ob_sample[:] = np.nan

# for object recognition
for mcount,mouse in enumerate(mice):
    sample = obrec_sample[mcount,2]
    test = obrec_test[mcount,2]
    if marray[mcount,1] == 2:
        tetarray_ob_sample[mcount] = sample
        tetarray_ob_test[mcount] = test
    if marray[mcount,1] == 1:
        conarray_ob_sample[mcount] = sample
        conarray_ob_test[mcount] = test

# for object location
for mcount,mouse in enumerate(mice):
    sample = obloc_sample[mcount,2]
    test = obloc_test[mcount,2]
    if marray[mcount,1] == 2:
        tetarray_loc_sample[mcount] = sample
        tetarray_loc_test[mcount] = test
    if marray[mcount,1] == 1:
        conarray_loc_sample[mcount] = sample
        conarray_loc_test[mcount] = test


tetarray_ob_sample = tetarray_ob_sample[~np.isnan(tetarray_ob_sample)]
tetarray_loc_sample = tetarray_loc_sample[~np.isnan(tetarray_loc_sample)]
conarray_ob_sample = conarray_ob_sample[~np.isnan(conarray_ob_sample)]
conarray_loc_sample = conarray_loc_sample[~np.isnan(conarray_loc_sample)]

tetarray_ob_test = tetarray_ob_test[~np.isnan(tetarray_ob_test)]
tetarray_loc_test = tetarray_loc_test[~np.isnan(tetarray_loc_test)]
conarray_ob_test = conarray_ob_test[~np.isnan(conarray_ob_test)]
conarray_loc_test = conarray_loc_test[~np.isnan(conarray_loc_test)]


objectscores_tet_sample = np.nanmean(tetarray_ob_sample[:])
locationscores_tet_sample = np.nanmean(tetarray_loc_sample[:])
sdobjectscores_tet_sample = np.nanstd(tetarray_ob_sample[:])/np.sqrt(8)
sdlocationscores_tet_sample = np.nanstd(tetarray_loc_sample[:])/np.sqrt(8)

objectscores_tet_test = np.nanmean(tetarray_ob_test[:])
locationscores_tet_test = np.nanmean(tetarray_loc_test[:])
sdobjectscores_tet_test = np.nanstd(tetarray_ob_test[:])/np.sqrt(8)
sdlocationscores_tet_test = np.nanstd(tetarray_loc_test[:])/np.sqrt(8)

objectscores_con_sample = np.nanmean(conarray_ob_sample[:])
locationscores_con_sample = np.nanmean(conarray_loc_sample[:])
sdobjectscores_con_sample = np.nanstd(conarray_ob_sample[:])/np.sqrt(8)
sdlocationscores_con_sample = np.nanstd(conarray_loc_sample[:])/np.sqrt(8)

objectscores_con_test = np.nanmean(conarray_ob_test[:])
locationscores_con_test = np.nanmean(conarray_loc_test[:])
sdobjectscores_con_test = np.nanstd(conarray_ob_test[:])/np.sqrt(8)
sdlocationscores_con_test = np.nanstd(conarray_loc_test[:])/np.sqrt(8)


objectrec_sample = [objectscores_con_sample,objectscores_tet_sample]
sdobjectrec_sample = [sdobjectscores_con_sample,sdobjectscores_tet_sample]
objectrec_test = [objectscores_con_test,objectscores_tet_test]
sdobjectrec_test = [sdobjectscores_con_test,sdobjectscores_tet_test]

locationrec_sample = [locationscores_con_sample,locationscores_tet_sample]
sdlocationrec_sample = [sdlocationscores_con_sample,sdlocationscores_tet_sample]
locationrec_test = [locationscores_con_test,locationscores_tet_test]
sdlocationrec_test = [sdlocationscores_con_test,sdlocationscores_tet_test]


# plot graph

index1 = np.hstack((1, 2))
n_groups = np.arange(3)
bar_width = 0.5


fig = plt.figure(figsize = (4,5.5))
ax = fig.add_subplot(111)

ax.plot(1,objectscores_con_sample, 'o', color = 'k')
ax.errorbar(1,objectscores_con_sample,sdobjectscores_con_sample, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,objectscores_tet_sample, 'o', color = 'k')
ax.errorbar(2,objectscores_tet_sample,sdobjectscores_tet_sample, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)

ax.plot(3,objectscores_con_test, 'o', color = 'k')
ax.errorbar(3,objectscores_con_test,sdobjectscores_con_test, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(4,objectscores_tet_test, 'o', color = 'k')
ax.errorbar(4,objectscores_tet_test,sdobjectscores_tet_test, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)

ax.plot(np.hstack((1,1,1,1,1,1,1,1)),conarray_ob_sample, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2,2,2)),tetarray_ob_sample, 'o', color = 'blue', alpha = 0.5, markersize = 10)

ax.plot(np.hstack((3,3,3,3,3,3,3,3)),conarray_ob_test, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((4,4,4,4,4,4,4,4)),tetarray_ob_test, 'o', color = 'blue', alpha = 0.5, markersize = 10)

ax.axvline(2.5,linewidth=2, color="black")
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
plt.locator_params(axis = 'x', nbins  = 7)
plt.locator_params(axis = 'y', nbins  = 7)
ax.axhline(0,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
ax.set_ylim(0,125)
ax.set_xlim(0.5,4.5)
plt.xticks(n_groups + bar_width, ('','','',''))
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)
fig.savefig('Plots/Figure6/TotalExp_ObjectRec_0100' +' .png', dpi = 200)
plt.close()

fig = plt.figure(figsize = (4,5.5))
ax = fig.add_subplot(111)

ax.plot(1,locationscores_con_sample, 'o', color = 'k')
ax.errorbar(1,locationscores_con_sample,sdlocationscores_con_sample, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,locationscores_tet_sample, 'o', color = 'k')
ax.errorbar(2,locationscores_tet_sample,sdlocationscores_tet_sample, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)

ax.plot(3,locationscores_con_test, 'o', color = 'k')
ax.errorbar(3,locationscores_con_test,sdlocationscores_con_test, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(4,locationscores_tet_test, 'o', color = 'k')
ax.errorbar(4,locationscores_tet_test,sdlocationscores_tet_test, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)

ax.plot(np.hstack((1,1,1,1,1,1,1,1)),conarray_loc_sample, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2,2,2)),tetarray_loc_sample, 'o', color = 'blue', alpha = 0.5, markersize = 10)

ax.plot(np.hstack((3,3,3,3,3,3,3,3)),conarray_loc_test, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((4,4,4,4,4,4,4,4)),tetarray_loc_test, 'o', color = 'blue', alpha = 0.5, markersize = 10)

ax.axvline(2.5,linewidth=2, color="black")
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =26)
plt.locator_params(axis = 'x', nbins  = 7)
plt.locator_params(axis = 'y', nbins  = 7)
ax.axhline(0,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
ax.set_ylim(0,125)
ax.set_xlim(0.5,4.5)
plt.xticks(n_groups + bar_width, ('','','',''))

plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)
fig.savefig('Plots/Figure6/TotalExp_ObjectLoc_0100' +' .png', dpi = 200)
plt.close()

