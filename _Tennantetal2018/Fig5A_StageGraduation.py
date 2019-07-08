# -*- coding: utf-8 -*-
"""

Two part script:
### 1. Calculates the proportion of mice that had graduated to stage 2 on each day of the experiment
### 2. Plots graduation day vs expression graphs


"""


# import packages and functions
from Functions_CoreFunctions_0100 import adjust_spines, makelegend2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

# Load data containing day of graduation to stage 2 for each mouse
marray = np.loadtxt('Data_Input/Behaviour_SummaryData/Stagegraduation_0100.txt',delimiter = '\t')

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,16.1)]# choose specific day/s
tetmice = ['M' + str(int(x)) for x in [2,5,6,7,8,9,11]]
conmice = ['M' + str(int(x)) for x in [1,3,4,10]]

#plot legend
def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="right", bbox_to_anchor=(0.85, 0.58), fontsize = "small", markerscale = 1)
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)


# calculate proportion of mice that have graduated for each training day
proportionday_store = np.zeros((len(days), 4))
n_groups = 2

for daycount,day in enumerate(marray,0):
    allmice = marray[daycount,:]
    tetmice = np.hstack((marray[daycount,1],marray[daycount,4],marray[daycount,5],marray[daycount,6],marray[daycount,7],marray[daycount,8],marray[daycount,10],marray[daycount,11],marray[daycount,12],marray[daycount,13]))
    conmice = np.hstack((marray[daycount,0],marray[daycount,2],marray[daycount,3],marray[daycount,9],marray[daycount,14],marray[daycount,15]))  
    htetmice = np.hstack((marray[daycount,1],marray[daycount,5],marray[daycount,6],marray[daycount,8]))
    ltetmice = np.hstack((marray[daycount,4],marray[daycount,7],marray[daycount,10],marray[daycount,11],marray[daycount,12],marray[daycount,13]))
    p_tet = (np.nansum(tetmice))/10
    p_htet = (np.nansum(htetmice))/4
    p_ltet = (np.nansum(ltetmice))/6
    p_con = (np.nansum(conmice))/6
    proportionday_store[daycount,0] = p_tet
    proportionday_store[daycount,1] = p_con
    proportionday_store[daycount,2] = p_htet
    proportionday_store[daycount,3] = p_ltet



# plot graph of proportion of graduation vs training day

daybins = np.arange(0,21.1,1) #number of days array
bar_width = 0.4
index = np.arange(n_groups)

"""
# plot proportion for GFP and TeLC mice

fig = plt.figure(figsize = (5,3))
ax = fig.add_subplot(111)
ax.plot(daybins,proportionday_store[:,0], color = '0.3', label = 'AAV-FLEX-TeLC', linewidth = 1.5)
ax.plot(daybins,proportionday_store[:,1], color = 'PaleVioletRed', label = 'AAV-FLEX-GFP',linewidth = 1.5)
makelegend3(fig,ax)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =12)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =12)
ax.set_ylabel('Proportion', fontsize=13, labelpad = 16)
ax.set_xlabel('Training day', fontsize=14, labelpad = 16)
ax.set_ylim(-0.2,1.05)
ax.set_xlim(1,20)
ax.axvline(1,linewidth=3, color="black")
ax.axhline(-0.2,linewidth=3, color="black")
plt.yticks(index, ('0','1'))
plt.yticks(rotation=70)
ax = plt.locator_params(axis = 'x', nbins = 4)
plt.subplots_adjust(hspace = 1, wspace = 0.5,  bottom = 0.3, left = 0.15, right = 0.72, top = .85)
fig.savefig('Plots/Figure5/Proportion_graduation_0200' +'.png', dpi = 200)
plt.close()
"""

# plot proportion for GFP and lTeLC and hTeLC mice

fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(111)
ax.plot(daybins,proportionday_store[:,3], color = 'blue', label = 'lTeLC', linewidth = 1.5)
ax.plot(daybins,proportionday_store[:,2], color = 'red', label = 'hTeLC', linewidth = 1.5)
ax.plot(daybins,proportionday_store[:,1], color = 'Black', label = 'GFP',linewidth = 1.5)
makelegend3(fig,ax)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =12)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =12)
ax.set_ylabel('Proportion', fontsize=13, labelpad = 16)
ax.set_xlabel('Training day', fontsize=14, labelpad = 16)
ax.set_ylim(-0.15,1.15)
ax.set_xlim(0,21)
ax.axvline(0,linewidth=3, color="black")
ax.axhline(-0.15,linewidth=3, color="black")
plt.yticks(index, ('0','1'))
plt.yticks(rotation=70)
ax = plt.locator_params(axis = 'x', nbins = 4)
plt.subplots_adjust(hspace = 1, wspace = 0.5,  bottom = 0.3, left = 0.15, right = 0.72, top = .85)
fig.savefig('Plots/Figure5/Proportion_graduation_0200' +'.png', dpi = 200)
plt.close()



# Store data

genotype = ("TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP" )
mouse = np.arange(1,16.1,1)
days = np.arange(1,22.1,1)

lTeLC = proportionday_store[:,3]
hTeLC = proportionday_store[:,2]
GFP = proportionday_store[:,1]

data = np.vstack((days, lTeLC, hTeLC, GFP))
data =  np.transpose(data)

np.savetxt('Data_Output/Figure5/Figure5_A_right_0100.csv', data,fmt =  '%s', delimiter = ',', header = 'Day, lTeLC, hTeLC, GFP ')



## Graduation days for all mice (taken from spreadsheet for data input reasons)

m1 = 7
m2 = 20
m3 = 9
m4 = 7
m5 = 15
m6 = 20
m7 = 20
m8 = 7
m9 = 20
m10 = 7
m11 = 8
m13 = 12
m14 = 11
m15 = 10
m16 = 7
m17 = 7

# stack days for tetanus and control mice
tet = np.hstack((m2,m5,m6,m7,m8,m9,m11,m13,m14,m15))
con = np.hstack((m1,m3,m4,m10,m16,m17))

htet = np.vstack((m2,m6,m7,m9))
ltet = np.vstack((m5,m8,m11,m13,m14,m15))

# average number of mice graduated per day
avg_tet = np.nanmean(tet)
sd_tet = np.nanstd(tet)/math.sqrt(10)
avg_con = np.nanmean(con)
sd_con = np.nanstd(con)/math.sqrt(6)

tt_total = np.hstack((avg_con,avg_tet))
tt_total_sd = np.hstack((sd_con,sd_tet))

hTeLC = np.nanmean(np.hstack((m2,m6,m7,m9)))
lTeLC = np.nanmean(np.hstack((m5,m8,m11,m13,m14,m15)))


# Plot vs expression

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



# Average expression

total = np.hstack((expressionstore[1,1],expressionstore[4,1],expressionstore[5,1],expressionstore[6,1],expressionstore[7,1],expressionstore[8,1],expressionstore[10,1],expressionstore[11,1],expressionstore[12,1],expressionstore[13,1]))
totalcon = np.hstack((expressionstore[0,1],expressionstore[2,1],expressionstore[3,1],expressionstore[9,1],expressionstore[14,1],expressionstore[15,1]))
total_probe = np.hstack((expressionstore[4,1],expressionstore[7,1],expressionstore[10,1],expressionstore[11,1],expressionstore[12,1],expressionstore[13,1]))
dorsal = np.hstack((expressionstore_d[1,1],expressionstore_d[4,1],expressionstore_d[5,1],expressionstore_d[6,1],expressionstore_d[7,1],expressionstore_d[8,1],expressionstore_d[10,1],expressionstore_d[11,1],expressionstore_d[12,1],expressionstore_d[13,1]))
dorsalcon = np.hstack((expressionstore_d[0,1],expressionstore_d[2,1],expressionstore_d[3,1],expressionstore_d[9,1],expressionstore_d[14,1],expressionstore_d[15,1]))
dorsal_probe = np.hstack((expressionstore_d[4,1],expressionstore_d[7,1],expressionstore_d[10,1],expressionstore_d[11,1],expressionstore_d[12,1],expressionstore_d[13,1]))
ventral = np.hstack((expressionstore_v[1,1],expressionstore_v[4,1],expressionstore_v[5,1],expressionstore_v[6,1],expressionstore_v[7,1],expressionstore_v[8,1],expressionstore_v[10,1],expressionstore_v[11,1],expressionstore_v[12,1],expressionstore_v[13,1]))
ventralcon = np.hstack((expressionstore_v[0,1],expressionstore_v[2,1],expressionstore_v[3,1],expressionstore_v[9,1],expressionstore_v[14,1],expressionstore_v[15,1]))
sdtotal = np.hstack((sdexpressionstore[1,1],sdexpressionstore[4,1],sdexpressionstore[5,1],sdexpressionstore[6,1],sdexpressionstore[7,1],sdexpressionstore[8,1],sdexpressionstore[10,1],sdexpressionstore[11,1],sdexpressionstore[12,1],sdexpressionstore[13,1]))
sdtotalcon = np.hstack((sdexpressionstore[0,1],sdexpressionstore[2,1],sdexpressionstore[3,1],sdexpressionstore[9,1],sdexpressionstore[14,1],sdexpressionstore[15,1]))
sdtotal_probe = np.hstack((sdexpressionstore[4,1],sdexpressionstore[7,1],sdexpressionstore[10,1],sdexpressionstore[11,1],sdexpressionstore[12,1],sdexpressionstore[13,1]))
sddorsal = np.hstack((sdexpressionstore_d[1,1],sdexpressionstore_d[4,1],sdexpressionstore_d[5,1],sdexpressionstore_d[6,1],sdexpressionstore_d[7,1],sdexpressionstore_d[8,1],sdexpressionstore_d[10,1],sdexpressionstore_d[11,1],sdexpressionstore_d[12,1],sdexpressionstore_d[13,1]))
sddorsalcon = np.hstack((sdexpressionstore_d[0,1],sdexpressionstore_d[2,1],sdexpressionstore_d[3,1],sdexpressionstore_d[9,1],sdexpressionstore_d[14,1],sdexpressionstore_d[15,1]))
sdventral = np.hstack((sdexpressionstore_v[1,1],sdexpressionstore_v[4,1],sdexpressionstore_v[5,1],sdexpressionstore_v[6,1],sdexpressionstore_v[7,1],sdexpressionstore_v[8,1],sdexpressionstore_v[10,1],sdexpressionstore_v[11,1],sdexpressionstore_v[12,1],sdexpressionstore_v[13,1]))
sdventralcon = np.hstack((sdexpressionstore_v[0,1],sdexpressionstore_v[2,1],sdexpressionstore_v[3,1],sdexpressionstore_v[9,1],sdexpressionstore_v[14,1],sdexpressionstore_v[15,1]))



# plot graduation day vs dorsal expression

slope,intercept,r_value, p_value, std_err = stats.linregress(dorsal,tet)
ablinevalues = []
for i in dorsal:
    ablinevalues.append(slope*i+intercept)

slope,intercept,r_value, p_value, std_err = stats.linregress(dorsalcon,con)
ablinevalues1 = []
for i in dorsalcon:
    ablinevalues1.append(slope*i+intercept)
print(dorsal)
fig = plt.figure(figsize = (5,4))
ax = fig.add_subplot(111)
ax.plot(dorsal, tet, 'o',color = 'red',markersize = 5,label = '   Low')
ax.errorbar(dorsal, tet, xerr=sddorsal, fmt='o', markersize = 5,color = 'red',ecolor = 'red')
ax.plot(dorsal,ablinevalues, '-',color = 'red', linewidth = 1)
ax.plot(dorsalcon, con, 'o',color = 'Black',markersize = 5,label = '   Low')
ax.errorbar(dorsalcon, con, xerr=sddorsalcon, fmt='o', markersize = 5,color = 'Black',ecolor = 'Black')
ax.plot(dorsalcon,ablinevalues1, '-',color = 'Black', linewidth = 1)
ax.set_xlim(-50,2000)
ax.set_ylim(-1,21)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =14)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 7, width = 2, labelsize =14)
plt.xticks(rotation=70)
plt.locator_params(axis = 'y', nbins  = 5)
plt.locator_params(axis = 'x', nbins  = 5)#plt.yticks(n_groups,('Ventral','','','Dorsal'))
ax.axhline(-1,linewidth=3, color="black")
ax.axvline(-50,linewidth=3, color="black")
ax.set_xlabel('Flourescence', fontsize=16, labelpad = 20)
ax.set_ylabel('G day', fontsize=16, labelpad = 20)
plt.subplots_adjust(hspace = 1, wspace = 0.5,  bottom = 0.3, left = 0.15, right = 0.72, top = .85)
fig.savefig('Plots/Figure5/GraduationVDorsalExpression' +'.png', dpi = 200)
plt.close()



# SAVE DATA FOR R

genotype = ("TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","TeLC","GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP" )
mouse = np.arange(1,16.1,1)
fluorescence = np.hstack((dorsal,dorsalcon))
gday = np.hstack((tet,con))
data = np.vstack((mouse,genotype,fluorescence,gday))
data = np.transpose(data)
np.savetxt('Data_Output/Graduation_dorsal.csv', data,delimiter = '\t', header = 'Mouse\tGenotype\tFlourescence\tGraduation', comments='',fmt='%s')

dfluorescence = np.hstack((dorsal,dorsalcon))
vfluorescence = np.hstack((ventral,ventralcon))


# SAVE DATA FOR MANUSCIPT

data = np.vstack((genotype, gday, dfluorescence, vfluorescence)); data = np.transpose(data)
np.savetxt('Data_Output/Figure5/Figure5_A_left_0100.csv', data,fmt =  '%s', delimiter = ',', header = 'Genotype, Graduation day, Dorsal Fluorescence, Ventral Fluorescence')


