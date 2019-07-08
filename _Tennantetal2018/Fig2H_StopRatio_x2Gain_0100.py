
"""


# Calculates ratio between stops in the visual reward zone and motor reward zone


"""


# import packages and functions
from Functions_Core_0100 import extractstops,filterstops,create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend, z_score1, adjust_spines, makelegend2,readhdfdata,maketrialarray, ratio_stop
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task18_0100.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(17,46.1)]
mice = ['M' + str(int(x)) for x in [6.1]]# specific day/s

# arrays for storing data
track1_b = np.zeros((len(mice), len(days)));track1_b[:,:] = np.nan
track1_nb = np.zeros((len(mice), len(days)));track1_nb[:,:] = np.nan
track1_p = np.zeros((len(mice), len(days)));track1_p[:,:] = np.nan
sdtrack1_b = np.zeros((len(mice), len(days)));track1_b[:,:] = np.nan
sdtrack1_nb = np.zeros((len(mice), len(days)));track1_nb[:,:] = np.nan
sdtrack1_p = np.zeros((len(mice), len(days)));track1_p[:,:] = np.nan


### --------------------------------------------------------------------------------------------------------------------- #

# some functions here slightly altered from whats in the functions_core_0100.py file to for increasing track length

def shuffle_analysis(stopsdata, trialids):
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = np.std(srbin, axis=0)                                 # Array(BINNR)
    
    return srbin_mean, srbin_std



def shuffle_analysis1(stopsdata, trialids, tracklength):
    # Calculate stop rate for each section of the track
    srbin = create_srdata1( stopsdata, trialids, tracklength )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = np.std(srbin, axis=0)                                 # Array(BINNR)
    
    return srbin_mean, srbin_std


# function to create histogram of stop locations vs trial
def create_srdata1( stops, trialids,tracklength ):
    if stops.size == 0:
        return np.zeros((tracklength,))

    # create histogram
    posrange = np.linspace(0, tracklength, num=tracklength+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1
    return H


### --------------------------------------------------------------------------------------------------------------------- #


# loop thorugh mice and days to get data
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing1...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        
        #define track length parameters
        rewardzonestart = saraharray[1,11]
        rewardzoneend = saraharray[1,12]
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        trial = np.max(saraharray[:,9])
        tracklength = np.max(saraharray[:,1])

        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != -10), 0)
        dailymouse = np.delete(saraharray, np.where(saraharray[:, 8] == 0), 0)
        dailymouse_nb = np.delete(dailymouse, np.where(dailymouse[:, 8] == -10), 0)

        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stops_b)
        stopsdata_nb = filterstops(stops_nb)
        stopsdata_p = filterstops(stops_p)

        if tracklength >22 and tracklength <28:
            tracklength = 26 
        try:
            if tracklength==26: #and first[0] > rewardzonestart-2:
                if dailymouse_b.size > 0:
                    trialids_b = np.unique(stopsdata_b[:, 2])
                    stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b)
                    ratio = ratio_stop(stops_f_b)
                    track1_b[mcount,dcount] = ratio
                if dailymouse_nb.size > 0:
                    trialids_nb = np.unique(stopsdata_nb[:, 2])
                    stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb)
                    ratio = ratio_stop(stops_f_nb)
                    track1_nb[mcount,dcount] =ratio
                if dailymouse_p.size > 0:
                    trialids_p = np.unique(stopsdata_p[:, 2])
                    stops_f_p, srbin_std = shuffle_analysis1( stopsdata_p, trialids_p, tracklength)
                    ratio = ratio_stop(stops_f_p)
                    track1_p[mcount,dcount] =ratio

        except IndexError:
            continue



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(17,46.1)]
mice = ['M' + str(int(x)) for x in [2,7,13]]# specific day/s

# arrays for storing data
track2_b = np.zeros((len(mice), len(days)));track2_b[:,:] = np.nan
track2_nb = np.zeros((len(mice), len(days)));track2_nb[:,:] = np.nan
track2_p = np.zeros((len(mice), len(days)));track2_p[:,:] = np.nan
sdtrack2_b = np.zeros((len(mice), len(days)));track2_b[:,:] = np.nan
sdtrack2_nb = np.zeros((len(mice), len(days)));track2_nb[:,:] = np.nan
sdtrack2_p = np.zeros((len(mice), len(days)));track2_p[:,:] = np.nan


# loop thorugh mice and days to get data
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue

        #define track length parameters
        rewardzonestart = saraharray[1,11]
        rewardzoneend = saraharray[1,12]
        trial = np.max(saraharray[:,9])
        tracklength = np.max(saraharray[:,1]) # max tracklength
        
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        # Extract data for beaconed, non-beaconed, probe
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != -10), 0)
        dailymouse = np.delete(saraharray, np.where(saraharray[:, 8] == 0), 0)
        dailymouse_nb = np.delete(dailymouse, np.where(dailymouse[:, 8] == -10), 0)

        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stops_b)
        stopsdata_nb = filterstops(stops_nb)
        stopsdata_p = filterstops(stops_p)

        if tracklength >22 and tracklength <27:
            tracklength = 26
        try:
            if trial >4 and tracklength==26: #and first[0] > rewardzonestart-2
                if dailymouse_b.size > 0:
                    trialids_b = np.unique(stopsdata_b[:, 2])
                    stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b)
                    ratio = ratio_stop(stops_f_b)
                    track2_b[mcount,dcount] =ratio
                
                if dailymouse_nb.size > 0:
                    trialids_nb = np.unique(stopsdata_nb[:, 2])
                    stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb)
                    ratio = ratio_stop(stops_f_nb)
                    track2_nb[mcount,dcount] =ratio
                if dailymouse_p.size > 0:
                    trialids_p = np.unique(stopsdata_p[:, 2])
                    stops_f_p, srbin_std = shuffle_analysis1( stopsdata_p, trialids_p, tracklength)
                    ratio = ratio_stop(stops_f_p)
                    track2_p[mcount,dcount] =ratio

        except IndexError:
            continue



# stack experiments then average
tracks1_b = np.nanmean(np.hstack((np.nanmean(track1_b, axis=1),np.nanmean(track2_b, axis=1))),axis=0)
tracks1_nb = np.nanmean(np.hstack((np.nanmean(track1_nb, axis=1),np.nanmean(track2_nb, axis=1))),axis=0)
tracks1_p = np.nanmean(np.hstack((np.nanmean(track1_p, axis=1),np.nanmean(track2_p, axis=1))),axis=0)

sdtracks1_b = np.nanstd(np.hstack((np.nanmean(track1_b, axis=1),np.nanmean(track2_b, axis=1))),axis=0)
sdtracks1_nb = np.nanstd(np.hstack((np.nanmean(track1_nb, axis=1),np.nanmean(track2_nb, axis=1))),axis=0)
sdtracks1_p = np.nanstd(np.hstack((np.nanmean(track1_p, axis=1),np.nanmean(track2_p, axis=1))),axis=0)

tracks1_b1 = np.hstack((np.nanmean(track1_b, axis=1),np.nanmean(track2_b, axis=1)))
tracks1_nb1 = np.hstack((np.nanmean(track1_nb, axis=1),np.nanmean(track2_nb, axis=1)))
tracks1_p1 = np.hstack((np.nanmean(track1_p, axis=1),np.nanmean(track2_p, axis=1)))



## PLOT MEANS

mice1 = np.hstack((tracks1_nb,tracks1_p))
mice1sd = np.hstack((sdtracks1_nb,sdtracks1_p))
index = np.hstack((1, 2))

n_groups = np.arange(3)
bar_width = 0.5
width = 0.4
z = np.arange(0,3,1)
X = n_groups+width/2

fig = plt.figure(figsize = (5,7))
ax = fig.add_subplot(111)
ax.plot(index,mice1, 'o', color = 'k')
ax.errorbar(index,mice1,mice1sd, fmt = 'o', color = 'k', capsize = 11, markersize = 16, elinewidth =5, capthick = 3)
ax.plot(np.hstack((1,1,1,1)),tracks1_nb1, 'o', color = 'k', alpha = 0.4, markersize = 13)
ax.plot(np.hstack((2,2,2,2)),tracks1_p1, 'o', color = 'k', alpha = 0.4, markersize = 13)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 14, width = 3, labelsize =22)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 14, width = 3, labelsize =22)
ax.set_ylabel('Ratio', fontsize=22, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(linewidth=6, color="black")
ax.axvline(0.5,linewidth=6, color="black")
ax.set_ylim(0,1)
plt.locator_params(axis = 'y', nbins  = 3)
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)
ax.set_xlim(0.5,2.5)
plt.subplots_adjust(hspace = 1, wspace = .5,  bottom = 0.25, left = 0.15, right = 0.9, top = .9)
fig.savefig('Plots/Figure2/Task18_StopRatio_Average_0100' +' .png', dpi = 200)
plt.close()



# save data for R

tracks1_nb1 = tracks1_nb1[~np.isnan(tracks1_nb1)]
tracks1_p1 = tracks1_p1[~np.isnan(tracks1_p1)]
trial = np.array([3,3,3,3,4,4,4,4])
mouse =np.array([1,2,3,4,1,2,3,4])
tracks = np.hstack((tracks1_nb1,tracks1_p1))
data = np.vstack((tracks,trial,mouse)); data = np.transpose(data)

np.savetxt('Data_Output/Figure2/Figure2_H_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Trial')


