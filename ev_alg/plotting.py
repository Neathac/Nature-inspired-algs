# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 

plt.figure(figsize=(12,8))
utils.plot_experiments('multi', ['default.ZDT6', 'weighted.ZDT6', 'adaptive.ZDT6', 'differential.ZDT6', 'differential-multi.ZDT6', 'hypervolume-weighted.ZDT6', 'hypervolume-default.ZDT6'])
plt.show()