#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:03:25 2020

@author: mlcv
"""

import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14,10]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist, Sequence
from pytracking.analysis.playback_results import playback_results

trackers = []


# 
# trackers.extend(trackerlist('TratCatAttn31-32', 'default2', range(0,1), 'TratCatAttn32_1'))
# trackers.extend(trackerlist('TratCatAttn31-32', 'default2', range(1,2), 'TratCatAttn32_2'))
# trackers.extend(trackerlist('TratCatAttn31-32', 'default2', range(2,3), 'TratCatAttn32_3'))
# trackers.extend(trackerlist('TratCatAttn31-32', 'default2', range(3,4), 'TratCatAttn32_4'))
# trackers.extend(trackerlist('TratCatAttn31-32', 'default2', range(4,5), 'TratCatAttn32_5'))

# trackers.extend(trackerlist('dimpFMFAttnAw', 'dimp50_49', range(1,2), 'dimp50_4_2'))
# trackers.extend(trackerlist('dimpFMFAttnAw', 'dimp50_49', range(2,3), 'dimp50_46_1'))

trackers.extend(trackerlist('RankingFast', 'default', range(1), 'RafnkinFast'))




filter_criteria = None
dataset = get_dataset('otb')
# sequence = [dataset['Basketball']]
# print_per_sequence_results(trackers, dataset, 'nfs', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
# playback_results(trackers, sequence[0])
 
# plot_results(trackers, dataset, 'uav', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
#               skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
print_results(trackers, dataset, 'otb', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

