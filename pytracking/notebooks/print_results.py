#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:03:25 2020

@author: mlcv
"""

import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist, Sequence
from pytracking.analysis.playback_results import playback_results

dataset = get_dataset('otb')

for i in range(50):
    trackers = []
    trackers.extend(trackerlist('dimpRank', 'dimpRank50_search', range(i,i+1), 'dimpRank50'))
    filter_criteria = None
    # sequence = [dataset['Basketball']]
    # print_per_sequence_results(trackers, dataset, 'otb', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
    # playback_results(trackers, sequence[0])
     
    # plot_results(trackers, dataset, 'otb', merge_results=True, plot_types=('success', 'prec'), 
    #               skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
    print("run_id:",i)
    print_results(trackers, dataset, 'otb', merge_results=True, plot_types=('success', 'prec'))

