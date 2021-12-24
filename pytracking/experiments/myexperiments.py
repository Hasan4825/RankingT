from pytracking.evaluation import Tracker, get_dataset, trackerlist
import os

def Ranking():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('Ranking', 'default', range(2))
    dataset = get_dataset('otb')
    return trackers, dataset
