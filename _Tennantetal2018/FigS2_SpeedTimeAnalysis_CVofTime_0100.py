# -*- coding: utf-8 -*-
"""

### Calculates the speed an animal runs along the track
- location bins are 10 cm 



"""

# IMPORT PACKAGES AND FUNCTIONS
from Functions_Core_0100 import extractrewards, makelegend,FirstStopTime, SplitTrials, SplitTrials2, maketrialarray, shuffle_analysis_pertrial3, z_score1, lowhighstops, filterstops, create_srdata, timetorz, extractstops, timetostop,adjust_spines, makelegend2,readhdfdata, create_timebindata
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
import random
from matplotlib import pyplot

# ------------------------------------------------------------------------------ #

#IMPORT DATA
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files

# Mice and days to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,27.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]

# SPECIFY TRACK PARAMETERS
REAL_LENGTH = 200 #track length in cm
HDF_LENGTH = 20 #track length in VU
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20 #location bins
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7 #stop threshold
bins = np.arange(0.5,20.5,1) # track bins

# ARRAYS FOR STORING DATA FOR ALL MICE ON ALL DAYS
m5stops = np.zeros((0,6))
timestore = np.zeros((0, 5))
quartiles = np.zeros((len(mice),4,2)); quartiles[:,:,:] = np.nan
var = np.zeros((len(mice),6))
allstops = np.zeros((len(mice), 20))


#GET AND STORE STOPS DATA
for mcount,mouse in enumerate(mice): # for each mouse
    stopsdata_p = np.zeros((0,13)) # empty array that has same number of columns as raw data
    max_trialno = 0
    # get data for all days
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data') # get raw datafile for mouse and day
        except KeyError:
            print ('Error, no file')
            continue
        print('##...', mcount,dcount, '...##')
        # get probe trial data for each day & stack
        trialarray = maketrialarray(saraharray) # make list of trial number per row in data
        saraharray[:,9] = trialarray[:,0] # replace trial number
        stopsdata = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0) # get just probe trials
        if stopsdata.size>0:
            # the point of the following is to stack days for individual mice. So need to replace trial numbers so it appears like one training day
            trialno =np.amax(stopsdata[:, 9]) # find max trial number of current session
            stopsdata[:, 9] += max_trialno
            max_trialno +=trialno
            stopsdata_p = np.vstack((stopsdata_p,stopsdata))
        dcount +=1

    if stopsdata_p.size >0:
        if mcount == 3 or mcount ==5 or mcount ==7 or mcount ==8:
            # get trials
            trialids = np.unique(stopsdata_p[:, 9])
            # for plotting stops against time
            timedata = timetostop(stopsdata_p, trialids)# add time from bb (replaces apsolute time)
            timedata = extractstops(timedata) # extract stops
            timedata = filterstops(timedata) # filter stops
            timestore = np.vstack((timestore, timedata)) # store data for all mice
            if mcount==3:
                time1 = timedata
            if mcount==5:
                time2 = timedata
            if mcount==6:
                time3 = timedata
            if mcount==7:
                time4 = timedata
            if mcount==8:
                time5 = timedata
            # for splitting data based on time to RZ
            trialids = np.unique(stopsdata_p[:, 9]) # get array of trial numbers
            dataarray = timetorz(stopsdata_p, trialids)# add time from bb to rz (replaces apsolute time)
            data = extractstops(dataarray) # extract stops
            datastore = filterstops(data) # filter stops
            #calculate cv for location and time
            mean = (np.nanmean(timedata[:,0]))-3; SD = np.nanstd(timedata[:,0]); CV = SD/mean
            var[mcount, 0] = CV; var[mcount, 1] = mean; var[mcount, 2] = SD
            mean = np.nanmean(timedata[:,1]); SD = np.nanstd(timedata[:,1]); CV = SD/mean
            var[mcount, 3] = CV; var[mcount, 4] = mean; var[mcount, 5] = SD
            #data for example plots
            if mcount == 5:
                m5 =datastore
            stops = create_srdata(datastore, trialids)
            avg_stops= np.nanmean(stops, axis=0)
            allstops[mcount,:] = avg_stops
    mcount +=1




# TASK 12

# IMPORT DATA
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# SPECIFY MICE
days = ['Day' + str(int(x)) for x in np.arange(15,27.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]# choose specific day/s

# Stores
quartiles1 = np.zeros((len(mice),4,2))
quartiles1[:,:,:] = np.nan
var1 = np.zeros((len(mice),6))
allstops1 = np.zeros((len(mice), 20))
timestore1 = np.zeros((0, 5))

# LOOP DAYS AND MICE
for mcount,mouse in enumerate(mice):
    max_trialno = 0
    stopsdata_p = np.zeros((0,13))
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        print('##...', mcount,dcount, '...##')
        
        trialarray = maketrialarray(saraharray) # make list of trial number per row in data
        saraharray[:,9] = trialarray[:,0] # replace trial number
        stopsdata = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0) # get just probe trials
        if stopsdata.size>0: # if there are probe trials
            # the point of the following is to stack days for individual mice. So need to replace trial numbers so it appears like one training day
            trialno =np.amax(stopsdata[:, 9]) # find max trial number
            stopsdata[:, 9] += max_trialno # add max trial number from previous session to trial numbers in current sessuon
            max_trialno +=trialno
            stopsdata_p = np.vstack((stopsdata_p,stopsdata)) #stack days
        dcount +=1

    if stopsdata_p.size >0:
        if mcount == 5 or mcount == 6 or mcount ==7:
            # get trials
            trialids = np.unique(stopsdata_p[:, 9])
            # for plotting stops against time
            timedata = timetostop(stopsdata_p, trialids)# add time from bb (replaces apsolute time)
            timedata = extractstops(timedata)
            timedata = filterstops(timedata)
            timestore1 = np.vstack((timestore1, timedata)) # store data for all mice
            if mcount==5:
                time6 = timedata
            if mcount==6:
                time7 = timedata
            if mcount==7:
                time8 = timedata

            # for splitting data based on time to RZ
            trialids = np.unique(stopsdata_p[:, 9])
            dataarray = timetorz(stopsdata_p, trialids)# add time from bb (replaces apsolute time)
            data = extractstops(dataarray) # extract stops
            datastore = filterstops(data) # filter stops
            #coeffient of variation of time
            mean = (np.nanmean(timedata[:,0]))-3; SD = np.nanstd(timedata[:,0]); CV = SD/mean
            var1[mcount, 0] = CV; var1[mcount, 1] = mean; var1[mcount, 2] = SD
            mean = np.nanmean(timedata[:,1]); SD = np.nanstd(timedata[:,1]); CV = SD/mean
            var1[mcount, 3] = CV; var1[mcount, 4] = mean; var1[mcount, 5] = SD
            stops = create_srdata(datastore, trialids)
            avg_stops= np.nanmean(stops, axis=0)
            allstops1[mcount,:] = avg_stops
    mcount +=1



#------------------------------------------------------------------------------------#


data = np.vstack((var[3,:], var[5,:], var1[5,:], var1[6,:], var1[7,:]))

np.savetxt('Data_Output/Supplemental2/FigureS2_CV_0100.csv', data,fmt = '%s', delimiter = ',', header = 'CV (location), Mean (Location), SD (Location), CV (Time), Mean (Time), SD (Time)')



# coefficient of variation for location
data = np.vstack((allstops[:,:],allstops1[:,:]))
var_loc = np.zeros((data.shape[1]))
for columncount, column in enumerate(data):
    mean = np.nanmean(data[columncount,:])
    variation = stats.variation(data[columncount,:])
    var_loc[columncount] = variation



timemin = np.amin(timestore[:,1])
timemax = np.amax(timestore[:,1])
trialids = np.unique(timestore[:,2])


trialids = np.unique(time2[:,2])
timebins2 = create_timebindata(time2, trialids)
trialids = np.unique(time4[:,2])
timebins4 = create_timebindata(time4, trialids)#timebins3 = np.histogram(time3[:,1], bins=100)
trialids = np.unique(time6[:,2])
timebins6 = create_timebindata(time6, trialids)#timebins3 = np.histogram(time3[:,1], bins=100)
trialids = np.unique(time7[:,2])
timebins7 = create_timebindata(time7, trialids)#timebins3 = np.histogram(time3[:,1], bins=100)
trialids = np.unique(time8[:,2])
timebins8 = create_timebindata(time8, trialids)#timebins3 = np.histogram(time3[:,1], bins=100)

timebins = np.nanmean(np.vstack((timebins2, timebins4, timebins6,timebins7, timebins8)), axis=0)
timebinssd = np.nanstd(np.vstack((timebins2, timebins4, timebins6,timebins7, timebins8)), axis=0)/math.sqrt(6)

timebins[timebins == 0] = 'nan'
timebinssd[timebinssd == 0] = 'nan'



bins = np.arange(0,100,1)
fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax = fig.add_subplot(1,3,2) #stops per trial
ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis

ax.plot(bins,timebins,'-',color = 'black',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.fill_between(bins,timebins-timebinssd,timebins+timebinssd, facecolor = 'black', alpha = 0.15)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_ylim(0,1)
ax.set_xscale('log')

adjust_spines(ax, ['left','bottom']) # removes top and right spines

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.3, left = 0.07, right = 0.82, top = 0.85)
fig.savefig('Plots/Supplemental2/Task13_Avgtime_Histogram' + '_0100.png',  dpi = 200)
plt.close()



# Figure S2B

high_con_probe = np.nanmean(np.vstack((allstops[:,:],allstops1[:,:])), axis = 0)
high_sd_con_probe = np.nanstd(np.vstack((allstops[:,:],allstops[:,:])), axis = 0)/math.sqrt(8)

bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax = fig.add_subplot(1,3,2) #stops per trial
ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis

ax.plot(bins,high_con_probe,'-',color = 'black',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.fill_between(bins,high_con_probe-high_sd_con_probe,high_con_probe+high_sd_con_probe, facecolor = 'black', alpha = 0.15)

ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,0.17)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.3, left = 0.07, right = 0.82, top = 0.75)
fig.savefig('Plots/Supplemental2/Task13_AllStop_Histogram' + '_0100.png',  dpi = 200)
plt.close()



