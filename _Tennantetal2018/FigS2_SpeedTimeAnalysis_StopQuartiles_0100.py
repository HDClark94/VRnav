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

days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]
bins = np.arange(0.5,20.5,1) # track bins

# SPECIFY TRACK PARAMETERS
REAL_LENGTH = 200 #track length in cm
HDF_LENGTH = 20 #track length in VU
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20 #location bins
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7 #stop threshold

# ARRAYS FOR STORING DATA FOR ALL MICE ON ALL DAYS
s2_con_high_beac = np.zeros((len(mice), len(bins),2));s2_con_high_nbeac = np.zeros((len(mice), len(bins), 2));s2_con_high_probe = np.zeros((len(mice), len(bins), 2))
s2_con_high_beac[:,:,:] = np.nan;s2_con_high_nbeac[:,:,:] = np.nan; s2_con_high_probe[:,:,:] = np.nan
s2_con_low_beac = np.zeros((len(mice), len(bins),2));s2_con_low_nbeac = np.zeros((len(mice), len(bins),2));s2_con_low_probe = np.zeros((len(mice), len(bins),2))
s2_con_low_beac[:,:,:] = np.nan;s2_con_low_nbeac[:,:,:] = np.nan; s2_con_low_probe[:,:,:] = np.nan

Z_high_beac = np.zeros((len(mice), len(bins),2));Z_high_nbeac = np.zeros((len(mice), len(bins),2));Z_high_probe = np.zeros((len(mice), len(bins),2))
Z_high_beac[:,:,:] = np.nan;Z_high_nbeac[:,:,:] = np.nan; Z_high_probe[:,:,:] = np.nan

Z_low_beac = np.zeros((len(mice), len(bins),2));Z_low_nbeac = np.zeros((len(mice), len(bins),2));Z_low_probe = np.zeros((len(mice), len(bins),2))
Z_low_beac[:,:,:] = np.nan;Z_low_nbeac[:,:,:] = np.nan; Z_low_probe[:,:,:] = np.nan

allstops = np.zeros((len(mice), 20))
m5stops = np.zeros((0,6))
timestore = np.zeros((0, 5))
quartiles = np.zeros((len(mice),4,2)); quartiles[:,:,:] = np.nan
var = np.zeros((len(mice),6))

# ------------------------------------------------------------------------------ #


#GET AND STORE STOPS DATA
for mcount,mouse in enumerate(mice):
    stopsdata_p = np.zeros((0,13))
    tm = 0
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data') # get raw datafile for mouse and day
        except KeyError:
            print ('Error, no file')
            continue
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        print('##...', mcount,dcount, '...##')
        # get probe trial data for each day & stack
        trialarray = maketrialarray(saraharray)
        saraharray[:,9] = trialarray[:,0]
        stopsdata = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        if stopsdata.size>0:
            trialno =np.amax(stopsdata[:, 9])
            stopsdata[:, 9] += tm
            tm +=trialno
            stopsdata_p = np.vstack((stopsdata_p,stopsdata))
        dcount +=1

    if stopsdata_p.size >0:
        if mcount == 3 or mcount ==5 or mcount == 6 or mcount == 7 or mcount ==8:
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
            #data for example plots
            if mcount == 5:
                m5 =datastore
            #find percentiles of times
            ut = np.unique(datastore[:,1]) # column 1 is time
            l = np.percentile(ut[:], 25)
            u = np.percentile(ut[:], 75)
            m = np.percentile(ut[:], 50)
            #    #split trials based on time
            low1 = np.delete(datastore, np.where(datastore[:, 1] > m), 0)
            high1 = np.delete(datastore, np.where(datastore[:, 1] < m), 0)
            lower_p = np.delete(low1, np.where(low1[:, 1] > l), 0)
            upper_p = np.delete(high1, np.where(high1[:, 1] < u), 0)
            low_p = np.delete(low1, np.where(low1[:, 1] < l), 0)
            high_p = np.delete(high1, np.where(high1[:, 1] > u), 0)
            # get trial numbers for each of the quartiles
            h_trials_p = np.unique(high_p[:,2])
            l_trials_p = np.unique(low_p[:,2])
            h2_trials_p = np.unique(upper_p[:,2])
            l2_trials_p = np.unique(lower_p[:,2])
            #calculate average stops
            high_stops_p = create_srdata(high_p, h_trials_p)
            avg_high_stops_p = np.nanmean(high_stops_p, axis=0)
            low_stops_p = create_srdata(low_p, l_trials_p)
            avg_low_stops_p = np.nanmean(low_stops_p, axis=0)
            upper_stops_p = create_srdata(upper_p, h2_trials_p)
            avg_upper_stops_p = np.nanmean(upper_stops_p, axis=0)
            lower_stops_p = create_srdata(lower_p, l2_trials_p)
            avg_lower_stops_p = np.nanmean(lower_stops_p, axis=0)
            #store average stops
            s2_con_high_probe[mcount,:,0] = avg_high_stops_p
            s2_con_low_probe[mcount,:,0] = avg_low_stops_p
            s2_con_high_probe[mcount,:,1] = avg_upper_stops_p
            s2_con_low_probe[mcount,:,1] = avg_lower_stops_p
            stops = create_srdata(datastore, trialids)
            avg_stops= np.nanmean(stops, axis=0)
            allstops[mcount,:] = avg_stops
            #calculate zscores
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(high_p, h_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_high_probe[mcount,:,0] = zscore_p
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( low_p, l_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_low_probe[mcount,:,0] = zscore_p
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( upper_p, h2_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_high_probe[mcount,:,1] = zscore_p
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( lower_p, l2_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_low_probe[mcount,:,1] = zscore_p

            # find preferred stopping location & time to RZ for each quartile
            upper = (np.argmax(avg_upper_stops_p[3:17]))+3
            high = (np.argmax(avg_high_stops_p[3:17]))+3
            low = (np.argmax(avg_low_stops_p[3:17]))+3
            lower = (np.argmax(avg_lower_stops_p[3:17]))+3
            h = np.median(np.unique(high_p[:,1]), axis=0)
            l = np.median(np.unique(low_p[:,1]), axis=0)
            up = np.median(np.unique(upper_p[:,1]), axis=0)
            lr = np.median(np.unique(lower_p[:,1]), axis=0)
            quartiles[mcount, 0,0] = upper
            quartiles[mcount, 0,1] = up
            quartiles[mcount, 1,0] = high
            quartiles[mcount, 1,1] = h
            quartiles[mcount, 2,0] = low
            quartiles[mcount, 2,1] = l
            quartiles[mcount, 3,0] = lower
            quartiles[mcount, 3,1] = lr

    mcount +=1




# TASK 12

# IMPORT DATA
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'
# SPECIFY MICE
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]# choose specific day/s
# Stores
s2_con_12_high_beac = np.zeros((len(mice), len(bins),2));s2_con_12_high_nbeac = np.zeros((len(mice), len(bins),2));s2_con_12_high_probe = np.zeros((len(mice), len(bins),2))
s2_con_12_high_beac[:,:,:] = np.nan;s2_con_12_high_nbeac[:,:,:] = np.nan;s2_con_12_high_probe[:,:,:] = np.nan
s2_con_12_low_beac = np.zeros((len(mice), len(bins),2));s2_con_12_low_nbeac = np.zeros((len(mice), len(bins),2));s2_con_12_low_probe = np.zeros((len(mice), len(bins),2))
s2_con_12_low_beac[:,:,:] = np.nan;s2_con_12_low_nbeac[:,:,:] = np.nan;s2_con_12_low_probe[:,:,:] = np.nan

Z_12_high_beac = np.zeros((len(mice), len(bins),2));Z_12_high_nbeac = np.zeros((len(mice), len(bins),2));Z_12_high_probe = np.zeros((len(mice), len(bins),2))
Z_12_high_beac[:,:,:] = np.nan;Z_12_high_nbeac[:,:,:] = np.nan; Z_12_high_probe[:,:,:] = np.nan

Z_12_low_beac = np.zeros((len(mice), len(bins),2));Z_12_low_nbeac = np.zeros((len(mice), len(bins),2));Z_12_low_probe = np.zeros((len(mice), len(bins),2))
Z_12_low_beac[:,:,:] = np.nan;Z_12_low_nbeac[:,:,:] = np.nan; Z_12_low_probe[:,:,:] = np.nan

allstops1 = np.zeros((len(mice), 20))

timestore1 = np.zeros((0, 5))
quartiles1 = np.zeros((len(mice),4,2))
quartiles1[:,:,:] = np.nan

var1 = np.zeros((len(mice),6))
# LOOP DAYS AND MICE
for mcount,mouse in enumerate(mice):
    tm = 0
    stopsdata_p = np.zeros((0,13))
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        print('##...', mcount,dcount, '...##')
        trialarray = maketrialarray(saraharray)
        saraharray[:,9] = trialarray[:,0]
        stopsdata = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        if stopsdata.size>0:
            trialno =np.amax(stopsdata[:, 9])
            stopsdata[:, 9] += tm
            tm +=trialno
            stopsdata_p = np.vstack((stopsdata_p,stopsdata))
        dcount +=1

    if stopsdata_p.size >0:
        if mcount == 5 or mcount == 6 or mcount ==7:
            # get trials
            trialids = np.unique(stopsdata_p[:, 9])
            # for plotting stops against time
            timedata = timetostop(stopsdata_p, trialids)# add time from bb (replaces apsolute time)
            timedata = extractstops(timedata)
            timedata = filterstops(timedata)
            timestore1 = np.vstack((timestore1, timedata))
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

            #find percentiles of times
            ut = np.unique(datastore[:,1])
            l = np.percentile(ut[:], 25)
            u = np.percentile(ut[:], 75)
            m = np.percentile(ut[:], 50)
            #    #split trials based on time
            low1 = np.delete(datastore, np.where(datastore[:, 1] > m), 0)
            high1 = np.delete(datastore, np.where(datastore[:, 1] < m), 0)
            lower_p = np.delete(low1, np.where(low1[:, 1] > l), 0)
            upper_p = np.delete(high1, np.where(high1[:, 1] < u), 0)
            low_p = np.delete(low1, np.where(low1[:, 1] < l), 0)
            high_p = np.delete(high1, np.where(high1[:, 1] > u), 0)
            # get trial numbers for each of the quartiles
            h_trials_p = np.unique(high_p[:,2])
            l_trials_p = np.unique(low_p[:,2])
            h2_trials_p = np.unique(upper_p[:,2])
            l2_trials_p = np.unique(lower_p[:,2])
            #calculate average stops
            high_stops_p = create_srdata(high_p, h_trials_p)
            avg_high_stops_p = np.nanmean(high_stops_p, axis=0)
            low_stops_p = create_srdata(low_p, l_trials_p)
            avg_low_stops_p = np.nanmean(low_stops_p, axis=0)
            upper_stops_p = create_srdata(upper_p, h2_trials_p)
            avg_upper_stops_p = np.nanmean(upper_stops_p, axis=0)
            lower_stops_p = create_srdata(lower_p, l2_trials_p)
            avg_lower_stops_p = np.nanmean(lower_stops_p, axis=0)
            #store average stops
            s2_con_12_high_probe[mcount,:,0] = avg_high_stops_p
            s2_con_12_low_probe[mcount,:,0] = avg_low_stops_p
            s2_con_12_high_probe[mcount,:,1] = avg_upper_stops_p
            s2_con_12_low_probe[mcount,:,1] = avg_lower_stops_p
            stops = create_srdata(datastore, trialids)
            avg_stops= np.nanmean(stops, axis=0)
            allstops1[mcount,:] = avg_stops

            #calculate zscores
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(high_p, h_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_12_high_probe[mcount,:,0] = zscore_p
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( low_p, l_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_12_low_probe[mcount,:,0] = zscore_p
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( upper_p, h2_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_12_high_probe[mcount,:,1] = zscore_p
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( lower_p, l2_trials_p )
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            Z_12_low_probe[mcount,:,1] = zscore_p
            
            # find preferred stopping location & time to RZ for each quartile
            upper = (np.argmax(avg_upper_stops_p[3:17]))+3
            high = (np.argmax(avg_high_stops_p[3:17]))+3
            low = (np.argmax(avg_low_stops_p[3:17]))+3
            lower = (np.argmax(avg_lower_stops_p[3:17]))+3
            
            h = np.median(np.unique(high_p[:,1]), axis=0)
            l = np.median(np.unique(low_p[:,1]), axis=0)
            up = np.median(np.unique(upper_p[:,1]), axis=0)
            lr = np.median(np.unique(lower_p[:,1]), axis=0)
            quartiles1[mcount, 0,0] = upper
            quartiles1[mcount, 0,1] = up
            quartiles1[mcount, 1,0] = high
            quartiles1[mcount, 1,1] = h
            quartiles1[mcount, 2,0] = low
            quartiles1[mcount, 2,1] = l
            quartiles1[mcount, 3,0] = lower
            quartiles1[mcount, 3,1] = lr



    mcount +=1



#------------------------------------------------------------------------------------#




data = m5[:,:2]
np.savetxt('Data_Output/Supplemental2/FigureS2_A_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Location (cm), Time to RZ')

## Figure S2 A1


# mouse 5 example data
#reorder data based on time to trial
m5[m5[:,1].argsort()]
fig = plt.figure(figsize = (12,3))

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axi
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(20-3, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.plot(m5[:,0],m5[:,1], 'o',color = 'Black', markersize = 5, label = 'Non - beaconed') #plot becaoned trials
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =18)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =18)
ax.set_xlim(0,20.5)
ax.set_ylim(0,33)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Supplemental2/Example' + 'Raster' + '_SpeedVSStop_' + '.png',  dpi = 200)
plt.close()



## Figure S2 A2

# average over days for all mice
high_con_probe = np.vstack((s2_con_high_probe[:,:,0],s2_con_12_high_probe[:,:,0]))
low_con_probe =np.vstack((s2_con_low_probe[:,:,0],s2_con_12_low_probe[:,:,0]))
upper_con_probe = np.vstack((s2_con_high_probe[:,:,1],s2_con_12_high_probe[:,:,1]))
lower_con_probe =np.vstack((s2_con_low_probe[:,:,1],s2_con_12_low_probe[:,:,1]))

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
ax.plot(bins,np.transpose(s2_con_high_probe[5,:,1]),color = 'red',label = 'Beaconed') #plot becaoned trials
ax.plot(bins,np.transpose(s2_con_high_probe[5,:,0]),color = 'blue',label = 'Beaconed') #plot becaoned trials
ax.plot(bins,np.transpose(s2_con_low_probe[5,:,0]),color = 'green',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.plot(bins,np.transpose(s2_con_low_probe[5,:,1]),color = 'black',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,1.1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Supplemental2/Task13_AvgStop_time_Histogram_mice4' + '_0100.png',  dpi = 200)
plt.close()







## Figure S2 B

# average over days for all mice
bins = np.arange(0.5,20.5,1)
fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax = fig.add_subplot(1,3,2) #stops per trial
ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-20, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,np.transpose(Z_high_probe[5,:,1]),color = 'red',label = 'Beaconed') #plot becaoned trials
ax.plot(bins,np.transpose(Z_high_probe[5,:,0]),color = 'blue',label = 'Beaconed') #plot becaoned trials
ax.plot(bins,np.transpose(Z_low_probe[5,:,0]),color = 'green',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.plot(bins,np.transpose(Z_low_probe[5,:,1]),color = 'black',label = 'Beaconed', alpha = 0.5) #plot becaoned
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Supplemental2/Task13_AvgZscore_time_Histogram_mice' + '_0100.png',  dpi = 200)
plt.close()



# average stops

high_con_probe = np.nanmean(np.vstack((s2_con_high_probe[:,:,0],s2_con_12_high_probe[:,:,0])), axis = 0)
high_sd_con_probe = np.nanstd(np.vstack((s2_con_high_probe[:,:,0],s2_con_12_high_probe[:,:,0])), axis = 0)/math.sqrt(8)
low_con_probe = np.nanmean(np.vstack((s2_con_low_probe[:,:,0],s2_con_12_low_probe[:,:,0])), axis = 0)
low_sd_con_probe = np.nanstd(np.vstack((s2_con_low_probe[:,:,0],s2_con_12_low_probe[:,:,0])), axis = 0)/math.sqrt(8)

upper_con_probe = np.nanmean(np.vstack((s2_con_high_probe[:,:,1],s2_con_12_high_probe[:,:,1])), axis = 0)
upper_sd_con_probe = np.nanstd(np.vstack((s2_con_high_probe[:,:,1],s2_con_12_high_probe[:,:,1])), axis = 0)/math.sqrt(8)
lower_con_probe = np.nanmean(np.vstack((s2_con_low_probe[:,:,1],s2_con_12_low_probe[:,:,1])), axis = 0)
lower_sd_con_probe = np.nanstd(np.vstack((s2_con_low_probe[:,:,1],s2_con_12_low_probe[:,:,1])), axis = 0)/math.sqrt(8)

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

ax.plot(bins,upper_con_probe,color = 'red',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.fill_between(bins,upper_con_probe-upper_sd_con_probe,upper_con_probe+high_sd_con_probe, facecolor = 'red', alpha = 0.15)
ax.plot(bins,high_con_probe,color = 'blue',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,high_con_probe-high_sd_con_probe,high_con_probe+high_sd_con_probe, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,low_con_probe,color = 'green',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,low_con_probe-low_sd_con_probe,low_con_probe+low_sd_con_probe, facecolor = 'green', alpha = 0.3)
ax.plot(bins,lower_con_probe,color = 'black',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.fill_between(bins,lower_con_probe-lower_sd_con_probe,lower_con_probe+lower_sd_con_probe, facecolor = 'black', alpha = 0.15)

ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,1.1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Supplemental2/Task13_AvgStop_time_Histogram' + '_0100.png',  dpi = 200)
plt.close()




# average over days for all mice
high_con_probe = np.nanmean(np.vstack((Z_high_probe[:,:,0],Z_12_high_probe[:,:,0])), axis = 0)
high_sd_con_probe = np.nanstd(np.vstack((Z_high_probe[:,:,0],Z_12_high_probe[:,:,0])), axis = 0)
low_con_probe = np.nanmean(np.vstack((Z_low_probe[:,:,0],Z_12_low_probe[:,:,0])), axis = 0)
low_sd_con_probe = np.nanstd(np.vstack((Z_low_probe[:,:,0],Z_12_low_probe[:,:,0])), axis = 0)

upper_con_probe = np.nanmean(np.vstack((Z_high_probe[:,:,1],Z_12_high_probe[:,:,1])), axis = 0)
upper_sd_con_probe = np.nanstd(np.vstack((Z_high_probe[:,:,1],Z_12_high_probe[:,:,1])), axis = 0)
lower_con_probe = np.nanmean(np.vstack((Z_low_probe[:,:,1],Z_12_low_probe[:,:,1])), axis = 0)
lower_sd_con_probe = np.nanstd(np.vstack((Z_low_probe[:,:,1],Z_12_low_probe[:,:,1])), axis = 0)


bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1) #
ax = fig.add_subplot(1,3,2) #
ax = fig.add_subplot(1,3,3) #
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-20, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,upper_con_probe,color = 'red',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.fill_between(bins,upper_con_probe-upper_sd_con_probe,upper_con_probe+high_sd_con_probe, facecolor = 'red', alpha = 0.15)
ax.plot(bins,high_con_probe,color = 'blue',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,high_con_probe-high_sd_con_probe,high_con_probe+high_sd_con_probe, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,low_con_probe,color = 'green',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,low_con_probe-low_sd_con_probe,low_con_probe+low_sd_con_probe, facecolor = 'green', alpha = 0.3)
ax.plot(bins,lower_con_probe,color = 'black',label = 'Beaconed', alpha = 0.5) #plot becaoned trials
ax.fill_between(bins,lower_con_probe-lower_sd_con_probe,lower_con_probe+lower_sd_con_probe, facecolor = 'black', alpha = 0.15)

ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])


plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Supplemental2/Task13_AvgZscore_time_Histogram' + '_0100.png',  dpi = 200)
plt.close()






