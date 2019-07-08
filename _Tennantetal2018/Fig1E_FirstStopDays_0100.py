# -*- coding: utf-8 -*-
"""

### Calculates the average first stop from the start of the track over training days


"""

# Import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, FirstStops, adjust_spines,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # Raw data

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)] # Days to analyse
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)] # Mice to analyse

# Arrays for storing data (output)
firststopstorebeac = np.zeros((len(days), len(mice)));firststopstorenbeac = np.zeros((len(days), len(mice)));firststopstoreprobe = np.zeros((len(days), len(mice)))
firststopstorebeac[:,:] = np.nan;firststopstorenbeac[:,:] = np.nan; firststopstoreprobe[:,:] = np.nan
sd_con_FirstStopsstorebeac = np.zeros((len(days), len(mice)));sd_con_FirstStopsstorenbeac = np.zeros((len(days), len(mice)));sd_con_FirstStopsstoreprobe = np.zeros((len(days), len(mice)))
sd_con_FirstStopsstorebeac[:,:] = np.nan;sd_con_FirstStopsstorenbeac[:,:] = np.nan; sd_con_FirstStopsstoreprobe[:,:] = np.nan


# For each day and mouse, pull raw data, calculate first stops and store data
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        print('##...', mcount,day, '...##')
        
        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*
        
        # get stops and trial arrays
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on         # get trial number
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p = extractstops(dailymouse_p)
        
        # Filter stops
        stops_b = filterstops(stops_b)
        stops_nb = filterstops(stops_nb)
        stops_p = filterstops(stops_p)
        
        # get first stop for each trial
        trarray = np.unique(trialarray[:,0]) # get unique trial numbers
        stops_f_b = FirstStops( trarray,stops_b ) # get locations of first stop for each trial
        beac = np.nanmean(stops_f_b, axis = 0) # find average first stop location
        sdbeac = np.nanstd(stops_f_b, axis = 0) # get sd of first stop location
        if stops_nb.size >0:
            stops_f_nb = FirstStops( trarray,stops_nb ) # get locations of first stop for each trial
            nbeac = np.nanmean(stops_f_nb, axis = 0) # find average first stop location
            sdnbeac = np.nanstd(stops_f_nb, axis = 0) # get sd of first stop location
        if stops_p.size >0 :
            stops_f_p = FirstStops( trarray,stops_p ) # get locations of first stop for each trial
            probe = np.nanmean(stops_f_p, axis = 0) # find average first stop location
            sdprobe = np.nanstd(stops_f_p, axis = 0) # get sd of first stop location

        # store data
        try:
            if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or mcount == 8:
                firststopstorebeac[dcount,mcount] = beac[0]*10 # x 10 to convert virtual units to cm
                sd_con_FirstStopsstorebeac[dcount,mcount] = beac[0]
                firststopstorenbeac[dcount, mcount] = nbeac[0]*10# x 10 to convert virtual units to cm
                sd_con_FirstStopsstorenbeac[dcount, mcount] = nbeac[0]
                if stops_p.size >0:
                    firststopstoreprobe[dcount, mcount] = probe[0]*10# x 10 to convert virtual units to cm
                    sd_con_FirstStopsstoreprobe[dcount, mcount] = probe[0]
        except IndexError:
            continue
        mcount +=1        



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]# choose specific day/s

# Arrays for storing data (output)
firststopstorebeac2 = np.zeros((len(days), len(mice)));firststopstorenbeac2= np.zeros((len(days), len(mice)));firststopstoreprobe2= np.zeros((len(days), len(mice)))
firststopstorebeac2[:,:] = np.nan;firststopstorenbeac2[:,:] = np.nan;firststopstoreprobe2[:,:] = np.nan
sd_con_12_FirstStopsstorebeac = np.zeros((len(days), len(mice)));sd_con_12_FirstStopsstorenbeac= np.zeros((len(days), len(mice)));sd_con_12_FirstStopsstoreprobe= np.zeros((len(days), len(mice)))
sd_con_12_FirstStopsstorebeac[:,:] = np.nan;sd_con_12_FirstStopsstorenbeac[:,:] = np.nan;sd_con_12_FirstStopsstoreprobe[:,:] = np.nan

# For each day and mouse, pull raw data, calculate first stops and store data
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        print('##...', mcount,day, '...##')

        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*
        
        # get stops and trial arrays
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on         # get trial number
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p = extractstops(dailymouse_p)
        
        # Filter stops
        stops_b = filterstops(stops_b)
        stops_nb = filterstops(stops_nb)
        stops_p = filterstops(stops_p)
        
        # get first stop for each trial
        trarray = np.unique(trialarray[:,0]) # get unique trial numbers
        stops_f_b = FirstStops( trarray,stops_b ) # get locations of first stop for each trial
        beac = np.nanmean(stops_f_b, axis = 0) # average first stop over trials
        sdbeac = np.nanstd(stops_f_b, axis = 0)# standard deviation of first stop location
        if stops_nb.size >0:
            stops_f_nb = FirstStops( trarray,stops_nb ) # get locations of first stop for each trial
            nbeac = np.nanmean(stops_f_nb, axis = 0)# average first stop over trials
            sdnbeac = np.nanstd(stops_f_nb, axis = 0)# standard deviation of first stop location
        if stops_p.size >0 :
            stops_f_p = FirstStops( trarray,stops_p ) # get locations of first stop for each trial
            probe = np.nanmean(stops_f_p, axis = 0)# average first stop over trials
            sdprobe = np.nanstd(stops_f_p, axis = 0)# standard deviation of first stop location

        try:
        # store data
            if mcount == 5 or mcount == 6 or mcount == 7:
                firststopstorebeac2[dcount,mcount] = beac[0]*10 # x 10 to convert virtual units to cm
                sd_con_12_FirstStopsstorebeac[dcount,mcount] = sdbeac[0]
                firststopstorenbeac2[dcount, mcount] = nbeac[0]*10# x 10 to convert virtual units to cm
                sd_con_12_FirstStopsstorenbeac[dcount, mcount] = sdnbeac[0]
                if stops_p.size >0:
                    firststopstoreprobe2[dcount, mcount] = probe[0]*10# x 10 to convert virtual units to cm
                    sd_con_12_FirstStopsstoreprobe[dcount, mcount] = sdprobe[0]
        except IndexError:
            continue
        mcount +=1        




# stack two experiments, then average

con_beac = np.nanmean(np.hstack((firststopstorebeac,firststopstorebeac2)), axis = 1)
con_nbeac = np.nanmean(np.hstack((firststopstorenbeac,firststopstorenbeac2)), axis =1)
con_probe = np.nanmean(np.hstack((firststopstoreprobe,firststopstoreprobe2)), axis = 1)
sd_con_beac = np.nanstd(np.hstack((firststopstorebeac,firststopstorebeac2)), axis = 1)/math.sqrt(8)
sd_con_nbeac = np.nanstd(np.hstack((firststopstorenbeac,firststopstorenbeac2)), axis =1)/math.sqrt(8)
sd_con_probe = np.nanstd(np.hstack((firststopstoreprobe,firststopstoreprobe2)), axis = 1)/math.sqrt(8)


# PLOT GRAPHS

n_groups = np.arange(0,22,1) # array of days
x = con_beac[0]

# plot average first stop over days for all mice
fig = plt.figure(figsize = (12,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvline(0, linewidth = 3, color = 'black')# bold line on the x axis
ax.axhline(30, linewidth = 3, color = 'black')# bold line on the x axis
ax.axhspan(88,100, linewidth = 0,facecolor='g', alpha=0.15, hatch = '/') # mark reward zone
ax.axhline(x,ls='--', color = 'black', linewidth = 1.5) # mark black box
ax.plot(n_groups, con_beac, 'o',color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
ax.errorbar(n_groups,con_beac,sd_con_beac, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15) # tick parameters: pad is tick label relative to
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
ax.set_xlim(0,22)
ax.set_ylim(30,100)
adjust_spines(ax, ['left','bottom']) # remove right and top axis
plt.locator_params(nbins=7, axis ='y') # set tick number on y axis
plt.locator_params(nbins=3, axis ='x') # set tick number on x axis
plt.xticks(rotation=70) # rotate x ticks 70 degrees
ax = plt.ylabel('Location (cm)', fontsize=16, labelpad = 18)


ax = fig.add_subplot(1,3,2)
ax.axvline(0, linewidth = 3, color = 'black')# bold line on the x axis
ax.axhline(30, linewidth = 3, color = 'black')# bold line on the y axis
ax.axhspan(88,100, linewidth = 0,facecolor='g', alpha=0.15, hatch = '/') # mark reward zone
ax.axhline(x, ls='--', color = 'black', linewidth = 1.5) # mark black box
ax.plot(n_groups, con_nbeac, 'o', color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
ax.errorbar(n_groups,con_nbeac,sd_con_nbeac, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
ax.set_xlim(0,22)
ax.set_ylim(30,100)
adjust_spines(ax, ['left','bottom']) # remove right and top axis
plt.locator_params(nbins=7, axis ='y') # set tick number on y axis
plt.locator_params(nbins=3, axis ='x') # set tick number on x axis
plt.xticks(rotation=70) # rotate x ticks 70 degrees

ax = fig.add_subplot(1,3,3)
ax.axvline(0, linewidth = 3, color = 'black')# bold line on the x axis
ax.axhline(30, linewidth = 3, color = 'black')# bold line on the x axis
ax.axhspan(88,100, linewidth = 0,facecolor='g', alpha=0.15, hatch = '/') # mark reward zone
ax.axhline(x, ls='--', color = 'black', linewidth = 1.5) # mark black box
ax.plot(n_groups, con_probe, 'o', color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
ax.errorbar(n_groups,con_probe,sd_con_probe, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
ax.set_xlim(0,22)
ax.set_ylim(30,100)
adjust_spines(ax, ['left','bottom']) # remove right and top axis
plt.locator_params(nbins=7, axis ='y') # set tick number on y axis
plt.locator_params(nbins=3, axis ='x') # set tick number on x axis
plt.xticks(rotation=70) # rotate x ticks 70 degrees

plt.figtext(0.08,0.85, 'Reward zone', fontsize = 14, fontstyle = 'italic') # mark reward zone
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure1/Task13_FirstStops_vsdays_0100'+'.png', dpi =200) # path to save figure to
plt.close()






# STORE DATA FOR R

x = np.hstack((firststopstorebeac[:,3],firststopstorebeac[:,5],firststopstorebeac[:,6],firststopstorebeac[:,7],firststopstorebeac[:,8],firststopstorebeac2[:,5],firststopstorebeac2[:,6],firststopstorebeac2[:,7]))
xsd = np.hstack((firststopstorebeac[:,3],firststopstorebeac[:,5],firststopstorebeac[:,6],firststopstorebeac[:,7],firststopstorebeac[:,8],firststopstorebeac2[:,5],firststopstorebeac2[:,6],firststopstorebeac2[:,7]))

x1 = np.hstack((firststopstorenbeac[:,3],firststopstorenbeac[:,5],firststopstorenbeac[:,6],firststopstorenbeac[:,7],firststopstorenbeac[:,8],firststopstorenbeac2[:,5],firststopstorenbeac2[:,6],firststopstorenbeac2[:,7]))
x1sd = np.hstack((firststopstorenbeac[:,3],firststopstorenbeac[:,5],firststopstorenbeac[:,6],firststopstorenbeac[:,7],firststopstorenbeac[:,8],firststopstorenbeac2[:,5],firststopstorenbeac2[:,6],firststopstorenbeac2[:,7]))

x2 = np.hstack((firststopstoreprobe[:,3],firststopstoreprobe[:,5],firststopstoreprobe[:,6],firststopstoreprobe[:,7],firststopstoreprobe[:,8],firststopstoreprobe2[:,5],firststopstoreprobe2[:,6],firststopstoreprobe2[:,7]))
x2sd = np.hstack((firststopstoreprobe[:,3],firststopstoreprobe[:,5],firststopstoreprobe[:,6],firststopstoreprobe[:,7],firststopstoreprobe[:,8],firststopstoreprobe2[:,5],firststopstoreprobe2[:,6],firststopstoreprobe2[:,7]))


days = np.arange(1,22.1,1)
days = np.hstack((np.hstack((days,days,days,days,days,days,days,days)), np.hstack((days,days,days,days,days,days,days,days)), np.hstack((days,days,days,days,days,days,days,days))))
m1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]); m8 = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
mice = np.hstack(([np.hstack((m1,m2,m3,m4,m5,m6,m7,m8)), np.hstack((m1,m2,m3,m4,m5,m6,m7,m8)), np.hstack((m1,m2,m3,m4,m5,m6,m7,m8))]))

b = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
nb = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
p = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])

trialtype = np.hstack((np.hstack((b,b,b,b,b,b,b,b)), np.hstack((nb,nb,nb,nb,nb,nb,nb,nb)), np.hstack((p,p,p,p,p,p,p,p))))
FirstStops = np.hstack((x,x1,x2))
data = np.vstack((mice, days, trialtype, FirstStops)); data = np.transpose(data)


np.savetxt('Data_Output/Figure1/Figure1_E_0100.csv', data,fmt =  '%i,%i,%i,%10.3f', delimiter = ',', header = 'Mouse, Day, Trialtype, Location (cm)')


