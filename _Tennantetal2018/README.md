# Tennantetal2018
Code used for behavioural analysis in Tennant et al, 2018 (Cell Reports)
Details on behaviour analysis for Tennant et al (2018)

# RAW DATA FOR BEHAVIOURAL EXPERIMENTS
Parsed raw data files for behavioural experiments (HDF5 format) can
be found under Data_Input>Behaviour_DataFiles.
HD5F format files:
The following information applies to all parsed raw datafiles used
for behavioural analysis for the Tennant et al (2018) paper in cell
reports.

Datafiles are organised as follows according to column number in raw
datafile:
0: Time from start of session (seconds)
1: Location (virtual units (1-20)) *
2: Speed (cm/second)
3: Speed (cm/second)
4: Reward (Yes/No) (1/0)
5: Licks (Yes/No) (1/0)
6: Start of reward zone (9)
7: End of reward zone (11)
8: Trial type (beaconed/non-beaconed/probe: 0/10/20) **
9: Trial number

# RAW DATA FOR BEHAVIOURAL EXPERIMENTS

# FIGURE 1
Experiment details:
Task 13:
Experimental mice: 1,2,3,5
Control mice: 4,6,7,8,9
Training days: 1-22
Task 12:
Experimental mice: 1,2,3,4,5
Control mice: 6,7,8
Training days: 1-22
Note: Experimental mice had no expression in Task13
Note: Experimental mice had expression in all Cre+ neurons in Task12
(i.e. in CA1, MEC L2, MEC L5, Sub)

# FIGURE 2
Experiment details:
Task 12:
Mice with 0.5x gain modulation: 1,4,6,7,8
Training days with gain modulation: 31-35 (mice are not staggered)
Task 18:
Mice with 2x gain modulation: 6
Training days with gain modulation: 15-46
Task 19:
Mice with 2x gain modulation: 2,7,13
Training days with gain modulation: 15-46 (mice are staggered)

# FIGURE 3
Experiment details:
Task 18:
Mice with increasing track lengths: 1,5
Training days with increasing track lengths: 0 - 20
Task 19:
Mice with increasing track lengths: 3,6,7,8,9
Training days with increasing track lengths: 1 - 46

# FIGURE 5
Experiment details:
Task 15
GFP expressing mice : 1,3,4,10
Mice with low TeLC expression: 5,8,10
Mice with high TeLC expression: 2,6,7,9
Training days: 1-22
Task 15b
GFP expressing mice : 4,5
Mice with low TeLC expression: 1,2,3
Training days: 1-19 ****

# FIGURE 6
See spreadsheet in RawData_ExplorationTasks for details of
experiment.
16 mice in total, all Sim1Cre, 8 males, 8 females. 8 with TeLC
expression, 8 with GFP expression - virus was counterbalanced
according to gender.

"""
# NOTES:
* There is an observed discrepancy between location along the
virtual track and location on the treadmill by approximately 2 cm.
I.e mouse appears in virtual space (in data, on computer when VR is
running) approx. 2cm more forward than on the treadmill (on the
torus screen). Therefore 0.2 virtual units (i.e. 2 cm) is added to
all location values.

** There is an error in trial incrimination in raw data for the
following experiments: Task 12, Task 13. In these datafiles, trial
number is incremented when the animal leaves the reward zone instead
of when the animal begins a new track/trial. To amend this, a short
function (maketrialarray) was written to rewrite the trial number
for each row in the HDF5 datafile. This function was tested
(Functions_Core_0100_test) and corrects the previous error in trial
incrimination.

*** Mouse became 1 ill after two training days into final training
week : was marked with a food deprivation score of 2, so showed mild
signs of food deprivation. Was fed some food before training but was
not able to perform in the task as usual. Therefore training days
after - are not analysed. This is mentioned in the methods of
Tennant et al, 2018.

**** Only training days till 19 are analysed for both Task 15 and
15b as task 15b ran for less days.
"""
