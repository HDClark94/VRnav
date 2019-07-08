# -*- coding: utf-8 -*-
"""


### Calculates the distance of the first stop from the start of the track


"""

# import packages and functions
from Functions_CoreFunctions_0100 import adjust_spines, makelegend2,readhdfdata
from Functions_Core_0100 import FirstStops, extractstops, maketrialarray, filterstops
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
import h5py
import matplotlib.gridspec as gridspec

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

nbins = np.arange(0,20+1e-6,0.1) # track bins
daybins = np.arange(0,19.1, 1) # days

# empty arrays for storing data
s2_con_firststopstorebeac = np.zeros((len(days), len(mice)));s2_con_firststopstorenbeac = np.zeros((len(days), len(mice)));s2_con_firststopstoreprobe = np.zeros((len(days), len(mice)))
s2_con_firststopstorebeac[:,:] = np.nan;s2_con_firststopstorenbeac[:,:] = np.nan; s2_con_firststopstoreprobe[:,:] = np.nan
s2_teth_firststopstorebeac = np.zeros((len(days), len(mice)));s2_teth_firststopstorenbeac = np.zeros((len(days), len(mice)));s2_teth_firststopstoreprobe = np.zeros((len(days), len(mice)))
s2_teth_firststopstorebeac[:,:] = np.nan;s2_teth_firststopstorenbeac[:,:] = np.nan; s2_teth_firststopstoreprobe[:,:] = np.nan
s2_tetl_firststopstorebeac = np.zeros((len(days), len(mice)));s2_tetl_firststopstorenbeac = np.zeros((len(days), len(mice)));s2_tetl_firststopstoreprobe = np.zeros((len(days), len(mice)))
s2_tetl_firststopstorebeac[:,:] = np.nan;s2_tetl_firststopstorenbeac[:,:] = np.nan; s2_tetl_firststopstoreprobe[:,:] = np.nan


# loop thorugh mice and days to get data
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # extract stops
        stops_b = extractstops(dailymouse_b)
        stops_p = extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stops_b)
        stops_p = filterstops(stops_p)
        
        # get trial number & average calculate first stop
        trialids_b = np.unique(stops_b[:, 2]) # get trial number
        if stops_p.size >0:
            trialids_p = np.unique(stops_p[:, 2]) # get trial number
        stops_f_b = FirstStops( trarray,stops_b ) # find location of first stop for every trial
        beac = np.nanmean(stops_f_b, axis = 0) # average the first stop across trials
        if stops_p.size >0:
            stops_f_p = FirstStops( trarray,stops_p ) # find location of first stop for every trial
            probe = np.nanmean(stops_f_p, axis = 0)# average the first stop across trials

        print('##...', mcount,day, '...##')
        # store data
        try:
            if trialids_p.size>0:
                if mcount == 2 or mcount == 3 or mcount ==9: # control mice
                    s2_con_firststopstorebeac[dcount,mcount] = beac[0]*10
                    if stops_p.size >0:
                        s2_con_firststopstoreprobe[dcount, mcount] = probe[0]*10
                if mcount == 0 and dcount <2: # control mice
                    s2_con_firststopstorebeac[dcount,mcount] = beac[0]*10
                    if stops_p.size >0:
                        s2_con_firststopstoreprobe[dcount, mcount] = probe[0]*10
                if mcount == 1 or mcount == 5 or mcount == 6 or mcount == 8: # high dorsal
                    s2_teth_firststopstorebeac[dcount,mcount] = beac[0]*10
                    if stops_p.size >0:
                        s2_teth_firststopstoreprobe[dcount, mcount] = probe[0]*10
                if mcount == 4 or mcount == 7 or mcount == 10:
                    s2_tetl_firststopstorebeac[dcount,mcount] = beac[0]*10
                    if stops_p.size >0:
                        s2_tetl_firststopstoreprobe[dcount, mcount] = probe[0]*10
        except IndexError:
            continue
        mcount +=1


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,5.1)]

# arrays for storing data (output)
s2_con_12_firststopstorebeac = np.zeros((len(days), len(mice)));s2_con_12_firststopstorenbeac= np.zeros((len(days), len(mice)));s2_con_12_firststopstoreprobe= np.zeros((len(days), len(mice)))
s2_con_12_firststopstorebeac[:,:] = np.nan;s2_con_12_firststopstorenbeac[:,:] = np.nan;s2_con_12_firststopstoreprobe[:,:] = np.nan
s2_teth_12_firststopstorebeac = np.zeros((len(days), len(mice)));s2_teth_12_firststopstorenbeac = np.zeros((len(days), len(mice)));s2_teth_12_firststopstoreprobe = np.zeros((len(days), len(mice)))
s2_teth_12_firststopstorebeac[:,:] = np.nan;s2_teth_12_firststopstorenbeac[:,:] = np.nan; s2_teth_12_firststopstoreprobe[:,:] = np.nan
s2_tetl_12_firststopstorebeac = np.zeros((len(days), len(mice)));s2_tetl_12_firststopstorenbeac = np.zeros((len(days), len(mice)));s2_tetl_12_firststopstoreprobe = np.zeros((len(days), len(mice)))
s2_tetl_12_firststopstorebeac[:,:] = np.nan;s2_tetl_12_firststopstorenbeac[:,:] = np.nan; s2_tetl_12_firststopstoreprobe[:,:] = np.nan

# loop thorugh mice and days to get data
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # extract stops
        stops_b = extractstops(dailymouse_b)
        stops_p = extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stops_b)
        stops_p = filterstops(stops_p)
        # get trial number
        trialids_b = np.unique(stops_b[:, 2]) # get trial number
        if stops_p.size >0:
            trialids_p = np.unique(stops_p[:, 2]) # find trial number
        stops_f_b = FirstStops( trarray,stops_b ) # find location of first stop for every trial
        beac = np.nanmean(stops_f_b, axis = 0)  # average the first stop across trials
        if stops_p.size >0:
            stops_f_p = FirstStops( trarray,stops_p )# find location of first stop for every trial
            probe = np.nanmean(stops_f_p, axis = 0)  # average the first stop across trials
        print('##...', mcount,day, '...##')
        #store data
        if mcount == 3 or mcount == 4:
            s2_con_12_firststopstorebeac[dcount,mcount] = beac[0]*10
            s2_con_12_firststopstoreprobe[dcount,mcount] = probe[0]*10

        if mcount == 0 or mcount == 1 or mcount == 2:
            s2_tetl_12_firststopstorebeac[dcount,mcount] = beac[0]*10
            try:
                if stops_p.size >0 and probe.size >0:
                    s2_tetl_12_firststopstoreprobe[dcount, mcount] = probe[0]*10
            except IndexError:
                print('Error!')

        mcount +=1

#------------------------------------------------------------------------------------#



# average over days for all mice
con_beac = np.nanmean(np.hstack((s2_con_firststopstorebeac,s2_con_12_firststopstorebeac)), axis = 0)
con_nbeac = np.nanmean(np.hstack((s2_con_firststopstorenbeac,s2_con_12_firststopstorenbeac)), axis =0)
con_probe = np.nanmean(np.hstack((s2_con_firststopstoreprobe,s2_con_12_firststopstoreprobe)), axis = 0)
sd_con_beac = np.nanstd(np.hstack((s2_con_firststopstorebeac,s2_con_12_firststopstorebeac)), axis = 0)/math.sqrt(8)
sd_con_nbeac = np.nanstd(np.hstack((s2_con_firststopstorenbeac,s2_con_12_firststopstorenbeac)), axis =0)/math.sqrt(8)
sd_con_probe = np.nanstd(np.hstack((s2_con_firststopstoreprobe,s2_con_12_firststopstoreprobe)), axis = 0)/math.sqrt(8)

teth_beac = np.nanmean(np.hstack((s2_teth_firststopstorebeac,s2_teth_12_firststopstorebeac)), axis = 0)
teth_nbeac = np.nanmean(np.hstack((s2_teth_firststopstorenbeac,s2_teth_12_firststopstorenbeac)), axis =0)
teth_probe = np.nanmean(np.hstack((s2_teth_firststopstoreprobe,s2_teth_12_firststopstoreprobe)), axis = 0)

tetl_beac = np.nanmean(np.hstack((s2_tetl_firststopstorebeac,s2_tetl_12_firststopstorebeac)), axis = 0)
tetl_nbeac = np.nanmean(np.hstack((s2_tetl_firststopstorenbeac,s2_tetl_12_firststopstorenbeac)), axis =0)
tetl_probe = np.nanmean(np.hstack((s2_tetl_firststopstoreprobe,s2_tetl_12_firststopstoreprobe)), axis = 0)


## calculate and plot means

con_beac = con_beac[~np.isnan(con_beac)]
con_nbeac = con_nbeac[~np.isnan(con_nbeac)]
con_probe = con_probe[~np.isnan(con_probe)]

teth_beac = teth_beac[~np.isnan(teth_beac)]
teth_nbeac = teth_nbeac[~np.isnan(teth_nbeac)]
teth_probe = teth_probe[~np.isnan(teth_probe)]

tetl_beac = tetl_beac[~np.isnan(tetl_beac)]
tetl_nbeac = tetl_nbeac[~np.isnan(tetl_nbeac)]
tetl_probe = tetl_probe[~np.isnan(tetl_probe)]

mice1 = [con_beac,tetl_beac,teth_beac,con_nbeac,tetl_nbeac,teth_nbeac,con_probe,tetl_probe,0]
mice1 = [con_beac,tetl_beac,teth_beac,con_probe,tetl_probe,0]

con_beacsd = np.std(con_beac)/math.sqrt(6)
con_probesd = np.std(con_probe)/math.sqrt(6)
tetl_beacsd = np.std(tetl_beac)/math.sqrt(6)
tetl_probesd = np.std(tetl_probe)/math.sqrt(6)
teth_beacsd = np.std(teth_beac)/math.sqrt(4)
con_beac1 = np.nanmean(con_beac)
con_probe1 = np.nanmean(con_probe)
tetl_beac1 = np.nanmean(tetl_beac)
tetl_probe1 = np.nanmean(tetl_probe)
teth_beac1 = np.nanmean(teth_beac)

mice1 = np.hstack((con_beac1,tetl_beac1,teth_beac1))
mice1sd = np.hstack((con_beacsd,tetl_beacsd,teth_beacsd))
mice2 = np.hstack((con_probe1,tetl_probe1))
mice2sd = np.hstack((con_probesd,tetl_probesd))



# plot graph

index = np.hstack((1, 2, 3))
index2 = np.hstack((1, 2))

n_groups = np.arange(3)
bar_width = 0.5
width = 0.4
z = np.arange(0,3,1)
X = n_groups+width/2

fig = plt.figure(1, figsize=(14,5))
gs = gridspec.GridSpec(1, 7)
ax = plt.subplot(gs[0, :3])
ax.plot(1,con_beac1, 'o', color = 'k')
ax.errorbar(1,con_beac1,con_beacsd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,tetl_beac1, 'o', color = 'blue')
ax.errorbar(2,tetl_beac1,tetl_beacsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(3,teth_beac1, 'o', color = 'red')
ax.errorbar(3,teth_beac1,teth_beacsd, fmt = 'o', color = 'red', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(np.hstack((1,1,1,1,1,1)),con_beac, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2)),tetl_beac, 'o', color = 'blue', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((3,3,3,3,)),teth_beac, 'o', color = 'red', alpha = 0.5, markersize = 10)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(30,90)
ax.set_xlim(0.5,3.5)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)

ax = plt.subplot(gs[0, 3:5])
ax.plot(1,con_probe1, 'o', color = 'k')
ax.errorbar(1,con_probe1,con_probesd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,tetl_probe1, 'o', color = 'blue')
ax.errorbar(2,tetl_probe1,tetl_probesd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(np.hstack((1,1,1,1,1,1)),con_probe, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2)),tetl_probe, 'o', color = 'blue', alpha = 0.5, markersize = 10)
#makelegend2(gs,ax)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
#ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(30,90)
ax.set_xlim(0.5,2.5)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)


ax = plt.subplot(gs[0, 5:7])
tetl = np.vstack((tetl_beac,tetl_probe))
tetlsd = np.hstack((tetl_beacsd,tetl_probesd))
tetl_beac1 = np.nanmean(tetl_beac)
tetl_probe1 = np.nanmean(tetl_probe)
tetl1 = np.hstack((tetl_beac1,tetl_probe1))
index1 = np.hstack((1, 2))
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.25, hatch = '/') # bold line on the x axis
ax.plot(index1,tetl1, 'o', color = 'blue', markersize = 14)
ax.errorbar(index1,tetl1,tetlsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(index1,tetl, '--o', color = 'blue', markersize = 10, linewidth = 2, alpha = 0.4) 
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 1)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.75,linewidth=3, color="black")
ax.set_xlim(0.75,2.25)
ax.set_ylim(30,90)

plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Plots/Figure5/Task15_FirstMeans_0100' +' .png', dpi = 200)
plt.close()


index = np.hstack((1, 2, 3))
index2 = np.hstack((1, 2))

n_groups = np.arange(3)
bar_width = 0.5
width = 0.4
z = np.arange(0,3,1)
X = n_groups+width/2

fig = plt.figure(figsize=(4,6))
#gs = gridspec.GridSpec(1, 7)
ax = fig.add_subplot(1,1,1)
ax.plot(1,con_beac1, 'o', color = 'k')
ax.errorbar(1,con_beac1,con_beacsd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,tetl_beac1, 'o', color = 'blue')
ax.errorbar(2,tetl_beac1,tetl_beacsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(3,teth_beac1, 'o', color = 'red')
ax.errorbar(3,teth_beac1,teth_beacsd, fmt = 'o', color = 'red', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(np.hstack((1,1,1,1,1,1)),con_beac, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2)),tetl_beac, 'o', color = 'blue', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((3,3,3,3,)),teth_beac, 'o', color = 'red', alpha = 0.5, markersize = 10)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(30,90)
ax.set_xlim(0.5,3.5)
plt.locator_params(axis = 'y', nbins  = 6)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Plots/Figure5/Task15_FirstMeans_Beaconed_0100' +' .png', dpi = 200)
plt.close()


fig = plt.figure(figsize=(4,6))
#gs = gridspec.GridSpec(1, 7)
ax = fig.add_subplot(1,1,1)
#ax = plt.subplot(gs[0, 3:5])
ax.plot(1,con_probe1, 'o', color = 'k')
ax.errorbar(1,con_probe1,con_probesd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,tetl_probe1, 'o', color = 'blue')
ax.errorbar(2,tetl_probe1,tetl_probesd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(np.hstack((1,1,1,1,1,1)),con_probe, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2)),tetl_probe, 'o', color = 'blue', alpha = 0.5, markersize = 10)
#makelegend2(gs,ax)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
#ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(30,90)
ax.set_xlim(0.5,2.5)
plt.locator_params(axis = 'y', nbins  = 6)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Plots/Figure5/Task15_FirstMeans_Probe_0100' +' .png', dpi = 200)
plt.close()


fig = plt.figure(figsize=(3,6))
#gs = gridspec.GridSpec(1, 7)
ax = fig.add_subplot(1,1,1)

tetl = np.vstack((tetl_beac,tetl_probe))
tetlsd = np.hstack((tetl_beacsd,tetl_probesd))
tetl_beac1 = np.nanmean(tetl_beac)
tetl_probe1 = np.nanmean(tetl_probe)
tetl1 = np.hstack((tetl_beac1,tetl_probe1))
index1 = np.hstack((1, 2))
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.25, hatch = '/') # bold line on the x axis
ax.plot(index1,tetl1, 'o', color = 'blue', markersize = 14)
ax.errorbar(index1,tetl1,tetlsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(index1,tetl, '--o', color = 'blue', markersize = 10, linewidth = 2, alpha = 0.4)

tetl = np.vstack((con_beac,con_probe))
tetlsd = np.hstack((con_beacsd,con_probesd))
tetl_beac1 = np.nanmean(con_beac)
tetl_probe1 = np.nanmean(con_probe)
tetl1 = np.hstack((con_beac1,con_probe1))
index1 = np.hstack((1, 2))
ax.plot(index1,tetl1, 'o', color = 'k', markersize = 14)
ax.errorbar(index1,tetl1,tetlsd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(index1,tetl, '--o', color = 'k', markersize = 10, linewidth = 2, alpha = 0.4)

adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.75,linewidth=3, color="black")
ax.set_xlim(0.75,2.25)
ax.set_ylim(30,90)

plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Plots/Figure5/Task15_FirstMeans_ComparisonlTeLC_0100' +' .png', dpi = 200)
plt.close()


fig = plt.figure(figsize=(3,6))
#gs = gridspec.GridSpec(1, 7)
ax = fig.add_subplot(1,1,1)

tetl = np.vstack((con_beac,con_probe))
tetlsd = np.hstack((con_beacsd,con_probesd))
tetl_beac1 = np.nanmean(con_beac)
tetl_probe1 = np.nanmean(con_probe)
tetl1 = np.hstack((con_beac1,con_probe1))
index1 = np.hstack((1, 2))
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.25, hatch = '/') # bold line on the x axis
ax.plot(index1,tetl1, 'o', color = 'blue', markersize = 14)
ax.errorbar(index1,tetl1,tetlsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(index1,tetl, '--o', color = 'blue', markersize = 10, linewidth = 2, alpha = 0.4)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhline(30,linewidth=3, color="black")
ax.axvline(0.75,linewidth=3, color="black")
ax.set_xlim(0.75,2.25)
ax.set_ylim(30,90)

plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Plots/Figure5/Task15_FirstMeans_ComparisonGFP_0100' +' .png', dpi = 200)
plt.close()


# -------------------------------------------------------------------------------------------------------------- #

# add expression data

mice = [str(int(x)) for x in np.arange(1,16.1)]
expression = np.loadtxt('Data_Input/ExpressionQuantification/T15_FinalQuantification_0300.txt')

expressionstore = np.zeros((len(mice), 3))
expressionstore_d = np.zeros((len(mice), 3))
expressionstore_v = np.zeros((len(mice), 3))
sdexpressionstore = np.zeros((len(mice), 3))
sdexpressionstore_d = np.zeros((len(mice), 3))
sdexpressionstore_v = np.zeros((len(mice), 3))

for mcount,mouse in enumerate(mice):
    marraybym = expression[expression[:,0]==mcount+1,:] # get data for each mouse
    rh = marraybym[:,1:4]
    lh = marraybym[:,4:]
    avg_RH = np.nanmean(rh[:,1])
    sd_RH = stats.sem(rh[:,1])
    avg_LH = np.nanmean(lh[:,1])
    sd_LH =stats.sem(lh[:,1])
    avg = np.nanmean(np.hstack((avg_RH,avg_LH)))
    sd = stats.sem(np.hstack((avg_RH,avg_LH)))
    expressionstore[mcount, 0] = avg
    expressionstore[mcount, 1] = avg_RH
    expressionstore[mcount, 2] = avg_LH
    sdexpressionstore[mcount, 0] = sd
    sdexpressionstore[mcount, 1] = sd_RH
    sdexpressionstore[mcount, 2] = sd_LH
    
    #for dorsal, mid and ventral
    rh_d = np.delete(rh, np.where(rh[:,2] != 1),0)
    rh_v = np.delete(rh, np.where(rh[:,2] != 2),0)
    lh_d = np.delete(lh, np.where(lh[:,2] != 1),0)
    lh_v = np.delete(lh, np.where(lh[:,2] != 2),0)
    
    avg_RH_d = np.nanmean(rh_d[:,1])
    avg_LH_d = np.nanmean(lh_d[:,1])
    avg_d = np.nanmean(np.hstack((avg_RH_d,avg_LH_d)))
    avg_RH_v = np.nanmean(rh_v[:,1])
    avg_LH_v = np.nanmean(lh_v[:,1])
    avg_v = np.nanmean(np.hstack((avg_RH_v,avg_LH_v)))
    
    sd_RH_d = stats.sem(rh_d[:,1])
    sd_LH_d = stats.sem(lh_d[:,1])
    sd_d = stats.sem(np.hstack((avg_RH_d,avg_LH_d)))
    sd_RH_v = stats.sem(rh_v[:,1])
    sd_LH_v = stats.sem(lh_v[:,1])
    sd_v = stats.sem(np.hstack((avg_RH_v,avg_LH_v)))
    
    expressionstore_d[mcount, 0] = avg_d
    expressionstore_d[mcount, 1] = avg_RH_d
    expressionstore_d[mcount, 2] = avg_LH_d
    expressionstore_v[mcount, 0] = avg_v
    expressionstore_v[mcount, 1] = avg_RH_v
    expressionstore_v[mcount, 2] = avg_LH_v
    
    sdexpressionstore_d[mcount, 0] = sd_d
    sdexpressionstore_d[mcount, 1] = sd_RH_d
    sdexpressionstore_d[mcount, 2] = sd_LH_d
    sdexpressionstore_v[mcount, 0] = sd_v
    sdexpressionstore_v[mcount, 1] = sd_RH_v
    sdexpressionstore_v[mcount, 2] = sd_LH_v
    
    mcount +=1



# save data for R

dorsal = np.hstack((expressionstore_d[0,1],expressionstore_d[2,1],expressionstore_d[3,1],expressionstore_d[9,1],expressionstore_d[14,1],expressionstore_d[15,1],expressionstore_d[4,1],expressionstore_d[7,1],expressionstore_d[10,1],expressionstore_d[11,1],expressionstore_d[12,1],expressionstore_d[13,1],expressionstore_d[1,1],expressionstore_d[5,1],expressionstore_d[6,1],expressionstore_d[8,1]))
print(dorsal)
teth_probe = np.zeros((4)); teth_probe[:] = np.nan
mice_b = np.hstack((con_beac,tetl_beac,teth_beac))
mice_p = np.hstack((con_probe,tetl_probe, teth_probe))
genotype = np.array(("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP" ,"lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","hTeLC","hTeLC","hTeLC","hTeLC"))

data = np.vstack((genotype, dorsal, mice_b, mice_p)); data=np.transpose(data)

np.savetxt('Data_Output/Figure5/Figure5_F_left_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Genotype, Dorsal Fluorescence,Beaconed,Probe')




