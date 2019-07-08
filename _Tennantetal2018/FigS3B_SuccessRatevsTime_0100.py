# -*- coding: utf-8 -*-
"""

@author: Sarah Tennant

# calculates the % of rewards mice recieve on different track lengths

"""

#IMPORT FUNCTIONS AND PACKAGES
from Functions_CoreFunctions_0100 import adjust_spines,readhdfdata, makelegend_small
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform

#Load data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task18_0100.h5'
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task18_FirstDays.txt',delimiter = '\t')

#specify mouse and days to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,20.1)]
mice = ['M' + str(int(x)) for x in [1,5]]# specific day/s

tracks = [str(int(x)) for x in np.arange(1,7.1)]

# empty arrays to store data
firststop_b = np.zeros((len(mice), 6, 2));firststop_b[:,:,:] = np.nan
firststop_nb = np.zeros((len(mice), 6, 2));firststop_nb[:,:,:] = np.nan
firststop_p = np.zeros((len(mice), 6, 2));firststop_p[:,:,:] = np.nan

# empty arrays to store data
time_b = np.zeros((len(mice), 6));firststop_b[:,:] = np.nan
time_nb = np.zeros((len(mice), 6));firststop_nb[:,:] = np.nan
time_p = np.zeros((len(mice), 6));firststop_p[:,:] = np.nan


# Input: array[:,13] (columns: for raw data, see README.py file)
# FUNCTION REPLACES ABSOLUTE TIME WITH TIME IT TAKES ANIMAL TO REACH REWARD ZONE
def timetorz(data, trialids,rewardzonestart):
    datastore = np.zeros((0,13)) # empty array same size as dataarray
    trials = np.zeros((trialids.shape[0],2)) # array(trialnumber,2)
    
    for trialcount, trial in enumerate(trialids):# loops through each trial
        tarray = data[data[:,9] ==trial,:] # get data only for each trial
        for row in tarray: # for every row of data
            if row[1] <=3.2: # if row is below black box
                starttime = row[0] # get time just after leaving black box
            if row[1] < rewardzonestart:
                rztime = float(row[0]) # get time just before leaving black box
            row+=1
        
        trialtime = rztime - starttime # time from black box to reward zone
        times = np.repeat(trialtime, tarray.shape[0]) # repeat trial time to same shape as trial data
        tarray[:,0] = times # replace time in tarray
        datastore =np.vstack((datastore,tarray)) # stack to empty array - after data from all trials added will end up with original dataset but time replaced
    return np.array(datastore)


# FOR EACH DAY AND MOUSE, PULL RAW DATA, CALCULATE STOPS PER TRIAL AND PLOT GRAPH
for mcount,mouse in enumerate(mice):
    daycount = 0
    for dcount,day in enumerate(days): #load mouse and day
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        length = np.max(saraharray[:,1]) # max track length
        trial = np.max(saraharray[:,9]) # max number of trials
        dayss = array[dcount,mcount] # array which has days to analyse (day with highest beaconed first stop on that track)
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        rewardzonestart = saraharray[1,11] # start of reward zne
        rewardzoneend = saraharray[1,12] # end of reward zone
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        binmin = np.min(saraharray[:,1]);binmax = np.max(saraharray[:,1]);interval = 0.1 # i.e if track is 20, 0.2 interval gives 100 bins

        # define tracklength according to reward zone (so its exact each time)
        if rewardzonestart == 8.8:
            tracklength = 20
            col = 0
        if rewardzonestart == 11.8:
            tracklength = 23
            col = 1
        if rewardzonestart == 16.3:
            tracklength = 29
            col = 2
        if rewardzonestart == 23.05:
            tracklength = 33.2
            col = 3
        if rewardzonestart == 33.1:
            tracklength = 43
            col = 4
        if rewardzonestart == 48.1:
            tracklength = 59.1
            col = 5
        bins = np.arange(0,tracklength/10,interval) # add 1e-6 so that last point included - array of bins for location

        # Extract data for beaconed, non-beaconed, probe
        if tracklength == 20:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks
        if tracklength == 23:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 30), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 40), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 50), 0)# delete all data not on probe tracks
        if tracklength == 29:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 60), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 70), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 80), 0)# delete all data not on probe tracks
        if tracklength == 33.2:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 43:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 59.1:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks

        # get array of trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        rewards_b = (np.sum(dailymouse_b[:,4])/trialids_b.size)*100
        rewards_nb = (np.sum(dailymouse_nb[:,4])/trialids_nb.size)*100
        rewards_p = (np.sum(dailymouse_p[:,4])/trialids_p.size)*100

        time_b = timetorz(dailymouse_b, trialids_b,rewardzonestart)
        time_nb = timetorz(dailymouse_nb, trialids_nb,rewardzonestart)
        time_p = timetorz(dailymouse_p, trialids_p,rewardzonestart)
        time_b = np.nanmean(time_b)
        time_nb = np.nanmean(time_nb)
        time_p = np.nanmean(time_p)
        try:
            if dayss == 1: # if its a day to analyse, store data
                if dailymouse_b.size > 0:
                    firststop_b[mcount,col,0] = rewards_b
                    firststop_b[mcount,col,1] = time_b
                if dailymouse_nb.size > 0:
                    firststop_nb[mcount,col,0] = rewards_nb
                    firststop_nb[mcount,col,1] = rewardzonestart*10
                if dailymouse_p.size > 0:
                    firststop_p[mcount,col,0] = rewards_p
                    firststop_p[mcount,col,1] = time_p
                daycount +=1
        except IndexError:
            continue



#Load data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task19_FirstDays.txt',delimiter = '\t')

#specify mouse and days to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,46.1)]
mice = ['M' + str(int(x)) for x in [3,6,7,8,9]]# specific day/s

tracks = [str(int(x)) for x in np.arange(1,7.1)] # number of tracks

# empty arrays to store data
firststop_b_2 = np.zeros((len(mice), 6, 2));firststop_b_2[:,:,:] = np.nan
firststop_nb_2 = np.zeros((len(mice), 6, 2));firststop_nb_2[:,:,:] = np.nan
firststop_p_2 = np.zeros((len(mice), 6, 2));firststop_p_2[:,:,:] = np.nan

time_b_2 = np.zeros((len(mice), 6));firststop_b_2[:,:] = np.nan
time_nb_2 = np.zeros((len(mice), 6));firststop_nb_2[:,:] = np.nan
time_p_2 = np.zeros((len(mice), 6));firststop_p_2[:,:] = np.nan

# FOR EACH DAY AND MOUSE, PULL RAW DATA, CALCULATE STOPS PER TRIAL AND PLOT GRAPH
for mcount,mouse in enumerate(mice):
    daycount = 0
    for dcount,day in enumerate(days): #load mouse and day
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        length = np.max(saraharray[:,1])
        trial = np.max(saraharray[:,9])
        dayss = array[dcount,mcount]
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        rewardzonestart = saraharray[1,11]
        rewardzoneend = saraharray[1,12]
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        binmin = np.min(saraharray[:,1]);binmax = np.max(saraharray[:,1]);interval = 0.1 # i.e if track is 20, 0.2 interval gives 100 bins
        bins = np.arange(0,tracklength/10,interval) # add 1e-6 so that last point included - array of bins for location
        
        # define track length
        if rewardzonestart == 8.8:
            tracklength = 20
            col = 0
        if rewardzonestart == 11.8:
            tracklength = 23
            col = 1
        if rewardzonestart == 16.3:
            tracklength = 29
            col = 2
        if rewardzonestart == 23.05:
            tracklength = 33.2
            col = 3
        if rewardzonestart == 33.1:
            tracklength = 43
            col = 4
        if rewardzonestart == 48.1:
            tracklength = 59.1
            col = 5

        # Extract data for beaconed, non-beaconed, probe
        if tracklength == 20:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks
        if tracklength == 23:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 30), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 40), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 50), 0)# delete all data not on probe tracks
        if tracklength == 29:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 60), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 70), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 80), 0)# delete all data not on probe tracks
        if tracklength == 33.2:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 43:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 59.1:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks
        
        # get array of trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        # calculate percent of rewards
        rewards_b = (np.sum(dailymouse_b[:,4])/trialids_b.size)*100
        rewards_nb = (np.sum(dailymouse_nb[:,4])/trialids_nb.size)*100
        rewards_p = (np.sum(dailymouse_p[:,4])/trialids_p.size)*100

        time_b = timetorz(dailymouse_b, trialids_b,rewardzonestart)
        time_nb = timetorz(dailymouse_b, trialids_b,rewardzonestart)
        time_p = timetorz(dailymouse_b, trialids_b,rewardzonestart)

        time_b = np.nanmean(time_b)
        time_nb = np.nanmean(time_nb)
        time_p = np.nanmean(time_p)
        
        try:
            if dayss == 1:
                if dailymouse_b.size > 0:
                    firststop_b_2[mcount,col,0] = rewards_b
                    firststop_b_2[mcount,col,1] = time_b
                if dailymouse_nb.size > 0:
                    firststop_nb_2[mcount,col,0] = rewards_nb
                    firststop_nb_2[mcount,col,1] = rewardzonestart*10
                if dailymouse_p.size > 0:
                    firststop_p_2[mcount,col,0] = rewards_p
                    firststop_p_2[mcount,col,1] = time_p
                daycount +=1
        except IndexError:
            continue





# stack experiments then average
tracks_b = np.nanmean(np.vstack((firststop_b,firststop_b_2)), axis=0)
tracks_nb = np.nanmean(np.vstack((firststop_nb, firststop_nb_2)), axis=0)
tracks_p = np.nanmean(np.vstack((firststop_p, firststop_p_2)), axis=0)
sdtracks_b = np.nanstd(np.vstack((firststop_b,firststop_b_2)), axis=0)/math.sqrt(6)
sdtracks_nb = np.nanstd(np.vstack((firststop_nb, firststop_nb_2)), axis=0)/math.sqrt(6)
sdtracks_p = np.nanstd(np.vstack((firststop_p, firststop_p_2)), axis=0 )/math.sqrt(6)

# plot graphs
n_groups = np.arange(7)
bar_width = 0.4
width = 0.4

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.plot(firststop_b[0,:,1],firststop_b[0,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb[0,:,1],firststop_nb[0,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p[0,:,1],firststop_p[0,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b[1,:,1],firststop_b[1,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb[1,:,1],firststop_nb[1,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p[1,:,1],firststop_p[1,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[0,:,1],firststop_b_2[0,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[0,:,1],firststop_nb_2[0,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[0,:,1],firststop_p_2[0,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[1,:,1],firststop_b_2[1,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[1,:,1],firststop_nb_2[1,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[1,:,1],firststop_p_2[1,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[2,:,1],firststop_b_2[2,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[2,:,1],firststop_nb_2[2,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[2,:,1],firststop_p_2[2,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[3,:,1],firststop_b_2[3,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[3,:,1],firststop_nb_2[3,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[3,:,1],firststop_p_2[3,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[4,:,1],firststop_b_2[4,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[4,:,1],firststop_nb_2[4,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[4,:,1],firststop_p_2[4,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.errorbar(tracks_b[:,1],tracks_b[:,0], sdtracks_b[:,0], fmt = '^', color = 'blue', capsize = 6, markersize = 20,elinewidth = 3, capthick = 3)
ax.errorbar(tracks_p[:,1],tracks_p[:,0],sdtracks_p[:,0], fmt = 's', color = 'red', capsize = 6, markersize = 15,elinewidth = 3, capthick = 3)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =20)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =18)
ax.set_ylabel('Rewards (%)', fontsize=20, labelpad = 18)
ax.set_xlabel('Time (s)', fontsize=20, labelpad = 18)
ax.axhline(-5,linewidth=3, color="black")
ax.axvline(0,linewidth=3, color="black")
ax.set_ylim(-5,110)
plt.locator_params(axis = 'y', nbins  = 4)
plt.locator_params(axis = 'x', nbins  = 6)

ax.set_xlim(0,40)

plt.subplots_adjust(hspace = 0.6, wspace = .5,  bottom = 0.3, left = 0.2, right = 0.8, top = .9)

fig.savefig('Plots/Supplemental3/Task18_SuccessRate_Tracks_time_log_0100' +' .png', dpi = 200)
plt.close()


# save data for R

tracks_b = np.vstack((firststop_b[:,:,0],firststop_b_2[:,:,0]))
tracks_nb = np.vstack((firststop_nb[:,:,0], firststop_nb_2[:,:,0]))
tracks_p = np.vstack((firststop_p[:,:,0], firststop_p_2[:,:,0]))

tracks_b = tracks_b.ravel()
tracks_nb = tracks_nb.ravel()
tracks_p = tracks_p.ravel()

"""
tracks = np.array((88,118,163,230,331,481))
tracks = np.hstack((tracks,tracks,tracks,tracks,tracks,tracks,tracks))
m1 = np.array([1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7])
mice = np.hstack((m1,m2,m3,m4,m5,m6,m7))
data = np.vstack((mice, tracks, tracks_b, tracks_nb, tracks_p)); data = np.transpose(data)


np.savetxt('Manuscript/Data/Figure2_J_time_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Reward zone,Beaconed, Non-beaconed,Probe')


"""





# PLOT GRADUATION DAY VS DORSAL EXPRESSION




tracks_p = np.vstack((firststop_p[:,:,1], firststop_p_2[:,:,1]))
tracks_p = tracks_p.ravel()
rewards_p = np.vstack((firststop_p[:,:,0], firststop_p_2[:,:,0]))
rewards_p = rewards_p.ravel()
tracks_p = tracks_p[~np.isnan(tracks_p)]
rewards_p = rewards_p[~np.isnan(rewards_p)]

print(firststop_p[:,0,1].shape, firststop_p_2[:,0,1].shape)
tracks1 = np.hstack((firststop_p[:,0,1], firststop_p_2[:,0,1]))
tracks1 = tracks1.ravel()
rewards1 = np.hstack((firststop_p[:,0,0], firststop_p_2[:,0,0]))
rewards1 = rewards1.ravel()
tracks1 = tracks1[~np.isnan(tracks1)]
rewards1 = rewards1[~np.isnan(rewards1)]


tracks2 = np.hstack((firststop_p[:,1,1], firststop_p_2[:,1,1]))
tracks2 = tracks2.ravel()
rewards2 = np.hstack((firststop_p[:,1,0], firststop_p_2[:,1,0]))
rewards2 = rewards2.ravel()
tracks2 = tracks2[~np.isnan(tracks2)]
rewards2 = rewards2[~np.isnan(rewards2)]

tracks3 = np.hstack((firststop_p[:,2,1], firststop_p_2[:,2,1]))
tracks3 = tracks3.ravel()
rewards3 = np.hstack((firststop_p[:,2,0], firststop_p_2[:,2,0]))
rewards3 = rewards3.ravel()
tracks3 = tracks3[~np.isnan(tracks3)]
rewards3 = rewards3[~np.isnan(rewards3)]

tracks4 = np.hstack((firststop_p[:,3,1], firststop_p_2[:,3,1]))
tracks4 = tracks4.ravel()
rewards4 = np.hstack((firststop_p[:,3,0], firststop_p_2[:,3,0]))
rewards4 = rewards4.ravel()
tracks4 = tracks4[~np.isnan(tracks4)]
rewards4 = rewards4[~np.isnan(rewards4)]

tracks5 = np.hstack((firststop_p[:,4,1], firststop_p_2[:,4,1]))
tracks5 = tracks5.ravel()
rewards5 = np.hstack((firststop_p[:,4,0], firststop_p_2[:,4,0]))
rewards5 = rewards5.ravel()
tracks5 = tracks5[~np.isnan(tracks5)]
rewards5 = rewards5[~np.isnan(rewards5)]

tracks6 = np.hstack((firststop_p[:,5,1], firststop_p_2[:,5,1]))
tracks6 = tracks6.ravel()
rewards6 = np.hstack((firststop_p[:,5,0], firststop_p_2[:,5,0]))
rewards6 = rewards6.ravel()
tracks6 = tracks6[~np.isnan(tracks6)]
rewards6 = rewards6[~np.isnan(rewards6)]


slope,intercept,r_value, p_value, std_err = stats.linregress(tracks_p,rewards_p)
ablinevalues = []
for i in tracks_p:
    ablinevalues.append(slope*i+intercept)


fig = plt.figure(figsize = (6,4))
ax = fig.add_subplot(111)

ax.plot(tracks_p, rewards_p, 'o',color = 'black',markersize = 4.5,label = '   Low')
ax.plot(tracks_p,ablinevalues, '-',color = 'Black', linewidth = 1)

ax.set_xlim(0,40)
ax.set_ylim(-2,102)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =14)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =14)
plt.xticks(rotation=70)
plt.locator_params(axis = 'y', nbins  = 5)
plt.locator_params(axis = 'x', nbins  = 5)#plt.yticks(n_groups,('Ventral','','','Dorsal'))
ax.axhline(-2,linewidth=3, color="black")
ax.axvline(0,linewidth=3, color="black")
ax.set_xlabel('Time (s)', fontsize=16, labelpad = 20)
ax.set_ylabel('Rewards (%)', fontsize=16, labelpad = 20)
plt.subplots_adjust(hspace = 1, wspace = 0.5,  bottom = 0.3, left = 0.15, right = 0.72, top = .85)
fig.savefig('Plots/Supplemental3/Timevsrewards' +'.png', dpi = 200)
plt.close()

print(r_value, 'r_value', p_value, 'p_value', slope,'slope')






slope,intercept,r_value, p_value, std_err = stats.linregress(tracks_p,rewards_p)
ablinevalues = []
for i in tracks_p:
    ablinevalues.append(slope*i+intercept)


fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(111)

ax.plot(tracks1, rewards1, 'o',color = 'black',markersize = 5,label = '   Track1')
ax.plot(tracks2, rewards2, 'o',color = 'blue',markersize = 5,label = '   Track2')
ax.plot(tracks3, rewards3, 'o',color = 'green',markersize = 5,label = '   Track3')
ax.plot(tracks4, rewards4, 'o',color = 'red',markersize = 5,label = '   Track4')
ax.plot(tracks5, rewards5, 'o',color = 'Violet',markersize = 5,label = '   Track5')
ax.plot(tracks6, rewards6, 'o',color = 'Orange',markersize = 5,label = '   Track6')
#ax.errorbar(tracks_p, rewards_p, xerr=sddorsal, fmt='o', markersize = 5,color = 'b',ecolor = 'b')
ax.plot(tracks_p,ablinevalues, '-',color = 'Black', linewidth = 1)
makelegend_small(fig,ax)
ax.set_xlim(5,35)
ax.set_ylim(-2,105)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =14)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =14)
plt.xticks(rotation=70)
plt.locator_params(axis = 'y', nbins  = 5)
plt.locator_params(axis = 'x', nbins  = 6)#plt.yticks(n_groups,('Ventral','','','Dorsal'))
ax.axhline(-2,linewidth=3, color="black")
ax.axvline(5,linewidth=3, color="black")
ax.set_xlabel('Time (s)', fontsize=16, labelpad = 20)
ax.set_ylabel('Rewards (%)', fontsize=16, labelpad = 20)
plt.subplots_adjust(hspace = 1, wspace = 0.5,  bottom = 0.3, left = 0.15, right = 0.75, top = .85)
fig.savefig('Plots/Supplemental3/Timevsrewards_0100' +'.png', dpi = 200)
plt.close()




tracks_b = np.vstack((firststop_b[:,:,1],firststop_b_2[:,:,1]))
tracks_nb = np.vstack((firststop_nb[:,:,1], firststop_nb_2[:,:,1]))
tracks_p = np.vstack((firststop_p[:,:,1], firststop_p_2[:,:,1]))

tracks_b = tracks_b.ravel()
tracks_nb = tracks_nb.ravel()
tracks_p = tracks_p.ravel()


tracks = np.array((88,118,163,230,331,481))
tracks = np.hstack((tracks,tracks,tracks,tracks,tracks,tracks,tracks))
m1 = np.array([1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7])
mice = np.hstack((m1,m2,m3,m4,m5,m6,m7))
data = np.vstack((mice, tracks, tracks_b, tracks_nb, tracks_p)); data = np.transpose(data)


np.savetxt('Data_Output/Supplemental3/timetorz_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Reward zone,Beaconed, Non-beaconed,Probe')







