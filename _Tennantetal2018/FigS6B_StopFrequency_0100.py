# -*- coding: utf-8 -*-
"""


### Calculates the location in which animals stop most frequently


"""

# import packages and functions
from Functions_Core_0100 import makelegend, maketrialarray, shuffle_analysis_pertrial3, z_score1, filterstops, create_srdata, timetorz, extractstops, timetostop, StopFrequency, adjust_spines, makelegend2,readhdfdata
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
import matplotlib.gridspec as gridspec


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

# ARRAYS FOR STORING DATA FOR ALL MICE ON ALL DAYS
control_b = np.zeros((len(mice)));control_b[:] = np.nan
tetl_b = np.zeros((len(mice)));tetl_b[:] = np.nan
teth_b = np.zeros((len(mice)));teth_b[:] = np.nan

control_nb = np.zeros((len(mice)));control_nb[:] = np.nan
tetl_nb = np.zeros((len(mice)));tetl_nb[:] = np.nan
teth_nb = np.zeros((len(mice)));teth_nb[:] = np.nan

control_p = np.zeros((len(mice)));control_p[:] = np.nan
tetl_p = np.zeros((len(mice)));tetl_p[:] = np.nan
teth_p = np.zeros((len(mice)));teth_p[:] = np.nan



# ------------------------------------------------------------------------------ #

#GET AND STORE STOPS DATA 

for mcount,mouse in enumerate(mice):
    stopsdata_b = np.zeros((0,5)); b=0
    stopsdata_p = np.zeros((0,5)); p=0
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        if mcount ==0 and dcount >0:
            break

        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # get trial number
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_p = extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stops_b)
        stops_p = filterstops(stops_p)
        
        #reassign trial number
        if dcount >0:
            trialno_b =np.amax(stops_b[:, 2])
            stops_b[:, 2] += b
            b +=trialno_b
            stopsdata_b = np.vstack((stopsdata_b, stops_b))
            if stops_p.size >0:
                trialno_p =np.amax(stops_p[:, 2])
                stops_p[:, 2] += p
                p +=trialno_p
                stopsdata_p = np.vstack((stopsdata_p, stops_p))
        dcount +=1

    trialids_b = np.unique(stops_b[:, 2])
    data_b = create_srdata( stops_b, trialids_b )
    beac = np.nansum(data_b, axis = 0)/trialids_b.shape[0]
    if stops_p.size>0:
        trialids_p = np.unique(stops_p[:, 2])
        # get stopping data
        data_p = create_srdata( stops_p, trialids_p )
        probe = np.nansum(data_p, axis = 0)/trialids_p.shape[0]

    print('##...', mcount,day, '...##')
    # store data
    if mcount == 0 or mcount == 2 or mcount == 3 or mcount == 9:
        loc = np.argmax(beac[3:17])
        control_b[mcount] = loc+3
        loc = np.argmax(probe[3:17])
        control_p[mcount] = loc+3
        print(loc, 'loc')
    if mcount == 1 or mcount == 5 or mcount == 8 or mcount == 6: # high dorsal
        loc = np.argmax(beac[3:17])
        teth_b[ mcount] = loc+3
        loc = np.argmax(probe[3:17])
        teth_p[ mcount] = loc+3
    if mcount == 7 or mcount == 10 or mcount == 4:
        loc = np.argmax(beac[3:17])
        tetl_b[ mcount] = loc+3
        loc = np.argmax(probe[3:17])
        tetl_p[ mcount] = loc+3

    mcount +=1


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,5.1)]# choose specific day/s

# Stores
control_b1 = np.zeros(( len(mice)));control_b1[:] = np.nan
tetl_b1 = np.zeros(( len(mice)));tetl_b1[:] = np.nan
control_nb1 = np.zeros(( len(mice)));control_nb1[:] = np.nan
tetl_nb1 = np.zeros(( len(mice)));tetl_nb1[:] = np.nan
control_p1 = np.zeros(( len(mice)));control_p1[:] = np.nan
tetl_p1 = np.zeros(( len(mice)));tetl_p1[:] = np.nan

for mcount,mouse in enumerate(mice):
    stopsdata_b = np.zeros((0,5)); b=0
    stopsdata_p = np.zeros((0,5)); nb=0
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # get trial number
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_p = extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stops_b)
        stops_p = filterstops(stops_p)
        
        #reassign trial number
        if dcount >0:
            trialno_b =np.amax(stops_b[:, 2])
            stops_b[:, 2] += b
            b +=trialno_b
            stopsdata_b = np.vstack((stopsdata_b, stops_b))
            #probe trials
            if stops_p.size >0:
                trialno_p =np.amax(stops_p[:, 2])
                stops_p[:, 2] += p
                p +=trialno_p
                stopsdata_p = np.vstack((stopsdata_p, stops_p))
        dcount+=1

    trialids_b = np.unique(stops_b[:, 2])
    data_b = create_srdata( stops_b, trialids_b )
    beac = np.nansum(data_b, axis = 0)/trialids_b.shape[0]
    if stops_p.size>0:
        trialids_p = np.unique(stops_p[:, 2])
        # get stopping data
        data_p = create_srdata( stops_p, trialids_p )
        probe = np.nansum(data_p, axis = 0)/trialids_p.shape[0]
    
    #STORE RATIOS
    if mcount == 3 or mcount == 4:
        loc = np.argmax(beac[3:17])
        control_b1[ mcount] = loc+3
        loc = np.argmax(probe[3:17])
        control_p1[ mcount] = loc+3
    if mcount == 0 or mcount == 1 or mcount == 2:
        loc = np.argmax(beac[3:17])
        tetl_b1[ mcount] = loc+3
        loc = np.argmax(probe[3:17])
        tetl_p1[ mcount] = loc

    mcount +=1



# average over days for all mice
con_beac = np.hstack((control_b,control_b1))
con_probe = np.hstack((control_p,control_p1))
sd_con_beac = np.nanstd(np.hstack((control_b,control_b1)), axis = 0)/math.sqrt(6)
sd_con_probe = np.nanstd(np.hstack((control_p,control_p1)), axis = 0)/math.sqrt(6)
tetl_beac = np.hstack((tetl_b,tetl_b1))
tetl_probe = np.hstack((tetl_p,tetl_p1))
sd_tetl_beac = np.nanstd(np.hstack((tetl_b,tetl_b1)), axis = 0)/math.sqrt(6)
sd_tetl_probe = np.nanstd(np.hstack((tetl_p,tetl_p1)), axis = 0)/math.sqrt(6)
teth_beac = teth_b
teth_probe = teth_p
sd_teth_beac = np.nanstd(teth_b, axis = 0)/math.sqrt(4)
sd_teth_probe = np.nanstd(teth_p, axis = 0)/math.sqrt(4)

con_beac = con_beac[~np.isnan(con_beac)]
con_probe = con_probe[~np.isnan(con_probe)]

teth_beac = teth_beac[~np.isnan(teth_beac)]
teth_probe = teth_probe[~np.isnan(teth_probe)]

tetl_beac = tetl_beac[~np.isnan(tetl_beac)]
tetl_probe = tetl_probe[~np.isnan(tetl_probe)]

mice1 = [con_beac,tetl_beac,teth_beac,con_probe,tetl_probe,0]
#np.transpose(days)
mice1 = [con_beac,tetl_beac,teth_beac,con_probe,tetl_probe,0]



## PLOT MEANS

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


index = np.hstack((1, 2, 3))
index2 = np.hstack((1, 2))

n_groups = np.arange(3)
bar_width = 0.5
width = 0.4
z = np.arange(0,3,1)
X = n_groups+width/2

fig = plt.figure(1, figsize=(10,5))

gs = gridspec.GridSpec(1, 7)

ax = plt.subplot(gs[0, :3])

ax.plot(1,con_beac1, 'o', color = 'k')
ax.errorbar(1,con_beac1,con_beacsd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,tetl_beac1, 'o', color = 'blue')
ax.errorbar(2,tetl_beac1,tetl_beacsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(3,teth_beac1, 'o', color = 'red')
ax.errorbar(3,teth_beac1,teth_beacsd, fmt = 'o', color = 'red', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
print('teth_beac',teth_beac.shape)
print(teth_b)
ax.plot(np.hstack((1,1,1,1,1,1)),con_beac, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2)),tetl_beac, 'o', color = 'blue', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((3,3,3,3)),teth_beac, 'o', color = 'red', alpha = 0.5, markersize = 10)

#makelegend2(gs,ax)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(0,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(0,20)
ax.set_xlim(0.5,3.5)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)

ax = plt.subplot(gs[0, 3:6])


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
ax.axhline(0,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(0,20)
ax.set_xlim(0.5,2.5)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)



plt.subplots_adjust(hspace = 1, wspace = 1,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)



plt.savefig('Plots/Supplemental6/Task15_StopFrequencyMeans_0100' +' .png', dpi = 200)
plt.close()











mice = [str(int(x)) for x in np.arange(1,16.1)]
expression = np.loadtxt('Data_Input/ExpressionQuantification/T15_FinalQuantification_0200.txt')

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


# Average expression

dorsal = np.hstack((expressionstore_d[0,1],expressionstore_d[2,1],expressionstore_d[3,1],expressionstore_d[9,1],expressionstore_d[14,1],expressionstore_d[15,1],expressionstore_d[4,1],expressionstore_d[7,1],expressionstore_d[10,1],expressionstore_d[11,1],expressionstore_d[12,1],expressionstore_d[13,1],expressionstore_d[1,1],expressionstore_d[5,1],expressionstore_d[6,1],expressionstore_d[8,1]))




teth_probe = np.zeros((4)); teth_probe[:] = np.nan
mice_b = np.hstack((con_beac,tetl_beac,teth_beac))
mice_p = np.hstack((con_probe,tetl_probe, teth_probe))
print('mice',mice_b.shape, mice_p.shape)
genotype = np.array(("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP" ,"lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","hTeLC","hTeLC","hTeLC","hTeLC"))
print('genotype',genotype.shape)

data = np.vstack((genotype, dorsal, mice_b, mice_p)); data=np.transpose(data)
print('x--', data.shape)
#data = np.hstack((data,x))
#print(data.shape)

np.savetxt('Data_Output/Supplemental6/FigureS6_B_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Genotype, Dorsal Fluorescence,Beaconed,Probe')




