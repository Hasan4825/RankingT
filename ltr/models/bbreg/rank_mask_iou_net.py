import torch.nn as nn
import torch, torchvision
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from pytracking.tracker.dimpRank.sample_generator import *
from torchvision.ops import roi_align

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class RankMaskIoUNet(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=(128,256), pred_input_dim=(256,256), pred_inter_dim=(256,256)):
        super().__init__()
        # _r for reference, _t for test
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.convs = conv(pred_input_dim[0]+pred_input_dim[1], 1, kernel_size=3, stride=1)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.

        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

        assert bb1.dim() == 3
        assert proposals2.dim() == 4

        num_images = proposals2.shape[0]
        self.num_sequences = proposals2.shape[1]

        # Extract first train sample
        feat1 = [f[0,...] if f.dim()==5 else f.reshape(-1, self.num_sequences, *f.shape[-3:])[0,...] for f in feat1]
        # bb1 = bb1[0,...]
        bb1 = bb1.reshape(-1,self.num_sequences,*bb1.shape[-2:])[0,...].reshape(-1,5).to(feat1[0].device)
        # Get modulation vector
        scores = self.get_modulation(feat1, bb1[:,:4])
        scores = torch.stack((scores.squeeze(), bb1[:,4]), dim=1)

        return scores

        
    def Sample_select(self, size, target_bbox):
        examples, ious = SampleGenerator('gaussian', size, 1.01, 1.02)(
        target_bbox.cpu().numpy(), 256, (0.1, 1.0))
        return  examples, ious

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (batch, 4)."""

        feat3_r, feat4_r = feat
        c3_r = self.conv3_1r(feat3_r)
        c4_r = self.conv4_1r(feat4_r)
        # Add batch_index to rois
        batch_size = bb.shape[0]*bb.shape[1]
        batch_index = (torch.arange(0,batch_size, dtype=torch.float32)//bb.shape[1]).reshape(-1,1).to(feat3_r.device)
        bb = bb.reshape(-1,4)
        roi1 = torch.cat((batch_index, bb), dim=1)
        roi3r = roi_align(c3_r, roi1, (3,3), spatial_scale=1/8)
        # roi3r = self.prroi_pool3r(c3_r, roi1)

        roi4r = roi_align(c4_r, roi1, (1,1), spatial_scale=1/16)

        # roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        # fc34_3_r = self.fc34_3r(fc34_r)
        scores = self.convs(fc34_r)
        # fc34_4_r = self.fc34_4r(fc34_r)

        return scores
    
   