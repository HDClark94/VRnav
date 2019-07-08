# VRnav
Code for post-data collection analysis of spatial navigation tasks captured using Oculus

Analysis of post collection data is split into two formats, native and legacy. 

Native format is the format outputted by the trial recording framework working in unity

Legacy format is an emmulation of the hdf5 files output used in Tennantetal2018 for behavioural plotting, To achieve legacy format, native must be passed through map2legacy and speed calculations ammended accordingly.

