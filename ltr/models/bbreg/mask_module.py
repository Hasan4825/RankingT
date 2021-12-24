import torch.nn as nn
import torch, torchvision
# from ltr.models.layers.blocks import LinearBlock
# from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
# from pytracking.tracker.dimpRank.sample_generator import *
import os
from torchvision.ops import roi_align
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNNHeads
from collections import OrderedDict
from torchvision.models.detection.roi_heads import paste_mask_in_image
from pytracking.evaluation.environment import env_settings



class Mask_Module(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim = 256, hidden_size=256, mask_net='Masknet.pth.tar', pretrained=True):
        super().__init__()
        self.mask_net = mask_net
        # _r for reference, _t for test
        self.conv1 = nn.Conv2d(512,256,1)
        self.conv2 = nn.Conv2d(1024,256,1)
        self.roi = MultiScaleRoIAlign(["layer2", "layer3"], 14, 2)
        # self.roi = MultiScaleRoIAlign(["layer3"], 14, 2)
        self.mask_head = MaskRCNNHeads(input_dim, [256, 256, 256, 256], 1)
        self.mask = MaskRCNNPredictor(input_dim, hidden_size, 1)
        self.image_shape = [(256,256)]
        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        #        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #        if m.bias is not None:
        #            m.bias.data.zero_()
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.uniform_()
        #        m.bias.data.zero_()
        
        if pretrained == True:
            mask_net_path = os.path.join(env_settings().network_path, self.mask_net)
            state_dict = torch.load(mask_net_path)
            state_dict = state_dict['net']
            keys = [key for key in state_dict.keys() if("mask_module" in key)]
            new_state_dict = OrderedDict()
            for key in keys:
                new_key = key[12:]
                new_state_dict[new_key] = state_dict[key]
            self.load_state_dict(new_state_dict)
        
        # if pretrained == True:
        #     state_dict = torch.load('/run/media/hasan/DATA/Hasan/pytracking/pytracking/networks/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth')
        #     keys = [key for key in state_dict.keys() if("mask_head" in key)]
        #     new_state_dict = OrderedDict()
        #     for key in keys:
        #         new_key = key[20:]
        #         new_state_dict[new_key] = state_dict[key]
        #     self.mask_head.load_state_dict(new_state_dict)
        #     keys = [key for key in state_dict.keys() if("mask_predictor" in key)]
        #     new_state_dict = OrderedDict()
        #     for key in keys:
        #         new_key = key[25:]
        #         if "conv5_mask" in new_key:
        #             new_state_dict[new_key] = state_dict[key]
        #     self.mask.load_state_dict(new_state_dict, strict=False)
            
    def forward(self, features, bboxes):
        feats= {}
        feats["layer2"] = self.conv1(features[0])
        feats["layer3"] = self.conv2(features[2])
        # feats["layer2"] = features[0]
        # feats["layer3"] = features[1]
        # roi3r = roi_align(feats['layer2'], bboxes, (14,14), spatial_scale=1/8)
        # roi4r = roi_align(feats['layer2'], bboxes, (14,14), spatial_scale=1/16)

        BBox = [x.reshape(-1,4) for x in bboxes]
        rois = self.roi(feats, BBox, self.image_shape)
        rois = self.mask_head(rois)
        mask = self.mask(rois)
        mask_prob = mask.sigmoid()
        # bboxes = torch.tensor(bboxes[0])
        # mask = paste_mask_in_image(mask_prob.squeeze(), bboxes[0].int(), self.image_shape[0][0], self.image_shape[0][1])
        return mask_prob, mask
            
            
            
# mask = Mask_Module()
# feat = {"layer2": torch.rand(1,512,36,36), "layer3": torch.rand(1,1024,18,18)}
# bb = torch.rand(1,4) * 256; bb[:, 2:] += bb[:, :2]
# msk = mask(feat, bb)

    
   