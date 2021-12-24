#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:03:25 2020

@author: mlcv
"""

import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 12]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist, Sequence
from pytracking.analysis.playback_results import playback_results

trackers = []





trackers.extend(trackerlist('ECO', 'default_deep', range(0,1), 'ECO'))
trackers.extend(trackerlist('UPDT', 'default', range(0,9), 'UPDT'))
trackers.extend(trackerlist('MDNet', 'default', range(0,1), 'MDNet'))
trackers.extend(trackerlist('CCOT', 'default', range(0,1), 'CCOT'))
# trackers.extend(trackerlist('DaSiamRPN', 'default', range(0,1), 'DaSiamRPN'))
# trackers.extend(trackerlist('SiamRPN++', 'default', range(0,1), 'SiamRPN++'))

trackers.extend(trackerlist('dimp', 'dimp50', range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimpFMFAttnAw', 'dimp50_49', range(1,2), 'TRAT'))
trackers.extend(trackerlist('TRAT', 'trat', range(0,3), 'TRAT'))
# trackers.extend(trackerlist('atom_R50_ours', 'default', range(0,5), 'ATOM50*'))
trackers.extend(trackerlist('atom', 'default', range(0,5), 'ATOM'))
# trackers.extend(trackerlist('siamban_results', 'SiamBAN_LaSOT', range(0,1), 'SiamBAN'))
# trackers.extend(trackerlist('Roam', 'LaSOT-ROAM++', range(0,1), 'Roam++'))
# trackers.extend(trackerlist('Roam', 'LaSOT-ROAM', range(0,1), 'Roam'))


filter_criteria = None
dataset = get_dataset('nfs')
# sequence = [dataset['book-10']]
# print_per_sequence_results(trackers, dataset, 'lasot', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)
# playback_results(trackers, sequence[0])
 
plot_results(trackers, dataset, 'nfs', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
# print_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

