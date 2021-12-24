from pytracking.utils import TrackerParams, FeatureParams, Choice
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features import deep, model
import torch

def parameters():
    params = TrackerParams()
    params.hard_rank = True
    params.gains = [1.0, 1.0]
    params.img_size = 107
    params.padding = 16
    params.batch_pos = 32
    params.batch_neg = 96
    params.batch_rank = 64
    params.batch_neg_cand= 1024
    params.batch_test= 256
    # candidates sampling
    params.n_samples= 256
    params.trans= 0.6
    params.scale= 1.05
    params.aspect_ratio= 1.0
    params.trans_limit= 1.5
    
    # training examples sampling
    params.trans_pos= 0.1
    params.scale_pos= 1.3
    params.trans_neg_init= 1
    params.scale_neg_init= 1.6
    params.trans_neg= 2
    params.scale_neg= 1.3
    params.trans_rank_up= 0.1
    params.scale_rank_up= 1.3
    params.trans_rank_down= 1
    params.scale_rank_down= 1.6
    # bounding box regression
    params.n_bbreg= 1000
    params.overlap_bbreg= [0.6, 1]
    params.trans_bbreg= 0.3
    params.scale_bbreg= 1.6
    params.aspect_bbreg= 1.1
    
    # initial training
    params.lr_init= 0.0005
    params.maxiter_init= 50
    params.n_pos_init= 500
    params.n_neg_init= 5000
    params.n_rank_init= 2000
    params.overlap_pos_init= [0.7, 1]
    params.overlap_neg_init= [0, 0.5]
    params.overlap_rank_init_up= [0.7, 1]
    params.overlap_rank_init_down= [0.06, 0.3]
    # online training
    params.lr_update= 0.001
    params.maxiter_update= 15
    params.n_pos_update= 50
    params.n_neg_update= 500
    params.n_rank_update= 100
    params.overlap_pos_update= [0.7, 1]
    params.overlap_neg_update= [0, 0.5]
    params.overlap_rank_update= [0.01, 1]
    params.overlap_rank_update_up= [0.7, 1]
    params.overlap_rank_update_down= [0.06, 0.3]
    # update criteria
    params.long_interval= 10
    params.n_frames_long= 100
    params.n_frames_short= 30

    # training 
    params.grad_clip= 10
    params.lr_mult= {'fc6': 10}
    params.ft_layers= ['fc']
    params.model = model.MDNet("/run/media/hasan/B709-D3C0/Hasan/pytrackingRanking/pytracking/networks/ranking_imagenet_all_125.pth")
    params.criterion = model.RankingLoss()
    params.model.set_learnable_params(params.ft_layers)
    params.init_optimizer = model.set_optimizer(params.model, params.lr_init, params.lr_mult)
    params.update_optimizer = model.set_optimizer(params.model, params.lr_update, params.lr_mult)
    # These are usually set from outside
    params.debug = 0                        # Debug level
    params.visualization = False            # Do visualization

    # Use GPU or not (IoUNet requires this to be True)
    params.use_gpu = True

    return params
