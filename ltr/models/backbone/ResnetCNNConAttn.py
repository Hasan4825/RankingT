import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from collections import OrderedDict
from ltr.models import backbone as backbones
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from torchvision.models.video.resnet import model_urls as model_url3d
from torch.hub import load_state_dict_from_url

import torchvision


class ResNet(nn.Module):

    def __init__(self, clip_len, output_layers, pretrained=True):
        super(ResNet, self).__init__()
        self.frame_num = clip_len
        self.Net2d = backbones.resnet.resnet50(pretrained=False)
        self.Net3d = backbones.resnet3d_cvpr.resnet50_3d(pretrained=False)
        self.output_layers = output_layers
        self.clip_len = clip_len
        self.eca0 = backbones.eca_net.eca_layerCon(2*(256 + 256), 5)
        self.eca1 = backbones.eca_net.eca_layerCon(2*(256 + 256 + 256 + 256), 5)
        self.frm = 0
    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

 
    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()
        self.frm += 1
        if output_layers is None:
            output_layers = self.output_layers
        if self.frame_num < self.clip_len: 
            x = x[:,:,-1,:,:]
            x = self.Net2d.conv1(x)
            x = self.Net2d.bn1(x)
            x = self.Net2d.relu(x)
    
            if self._add_output_and_check('conv1', x, outputs, output_layers):
                return outputs
    
            x = self.Net2d.maxpool(x)
    
            x = self.Net2d.layer1(x)
    
            if self._add_output_and_check('layer1', x, outputs, output_layers):
                return outputs
    
            x = self.Net2d.layer2(x)
    
            if self._add_output_and_check('layer2', x, outputs, output_layers):
                return outputs
    
            x = self.Net2d.layer3(x)
    
            if self._add_output_and_check('layer3', x, outputs, output_layers):
                return outputs
    
            x = self.Net2d.layer4(x)
    
            if self._add_output_and_check('layer4', x, outputs, output_layers):
                return outputs
    
            x = self.Net2d.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.Net2d.fc(x)
    
            if self._add_output_and_check('fc', x, outputs, output_layers):
                return outputs
    
            if len(output_layers) == 1 and output_layers[0] == 'default':
                return x
    
            raise ValueError('output_layer is wrong.')
            
        else:
            x_2d = x[:,:,-1,:,:]
            x_2d = self.Net2d.conv1(x_2d)
            x_2d = self.Net2d.bn1(x_2d)
            x_2d = self.Net2d.relu(x_2d)
            
            # x_3d = self.Net3d.stem(x)
            x_3d = self.Net3d.conv1(x)
            x_3d = self.Net3d.bn1(x_3d)
            x_3d = self.Net3d.relu(x_3d)
            if self._add_output_and_check('conv1', x, outputs, output_layers):
                return outputs
    
            x_2d = self.Net2d.maxpool(x_2d)
            x_3d = self.Net3d.maxpool(x_3d)

            x_2d = self.Net2d.layer1(x_2d)
            x_3d = self.Net3d.layer1(x_3d)

            if self._add_output_and_check('layer1', x, outputs, output_layers):
                return outputs

            x_2d = self.Net2d.layer2(x_2d)
            x_3d = self.Net3d.layer2(x_3d)
            # x_att_layer2 = torch.cat((x_2d, x_3d.squeeze(2)), dim=1)
            # x_att_layer2 = self.cfam2(x_att_layer2)
            # x_att_layer2 = self.attLayer2(x_att_layer2)
            x2_attn = self.eca0(torch.cat((x_2d, x_3d.squeeze(2)), dim=1))
            if self.frm >= 4:
                from matplotlib import pyplot as plt
                for i in range(len(x_3d[0])):
                    plt.imshow(x_2d[0,i].squeeze().cpu().detach().numpy()) 
                    plt.show()
            if self._add_output_and_check('layer2', x2_attn, outputs, output_layers):
                return outputs
            
            x_2d = self.Net2d.layer3(x_2d)
            x_3d = self.Net3d.layer3(x_3d)
            # x_att_layer3 = torch.cat((x_2d, x_3d.squeeze(2)), dim=1)
            # x_att_layer3 = self.cfam3(x_att_layer3)     
            # x_att_layer3 = self.attLayer3(x_att_layer3)
            x3_attn = self.eca1(torch.cat((x_2d, x_3d.squeeze(2)), dim=1))
            if self._add_output_and_check('layer3', x3_attn, outputs, output_layers):
                return outputs

            x_2d= self.Net2d.layer4(x_2d)
            x_3d = self.Net3d.layer4(x_3d)

            if self._add_output_and_check('layer4', x, outputs, output_layers):
                return outputs

            x_2d = self.Net2d.avgpool(x_2d)
            x_3d = self.Net3d.avgpool(x_3d)
            x_2d = x_2d.view(x_2d.size(0), -1)
            x_3d = x_3d.view(x_3d.size(0), -1)
            x_2d = self.Net2d.fc(x_2d)
            x_3d = self.Net3d.fc(x_3d)

            if self._add_output_and_check('fc', x, outputs, output_layers):
                return outputs
    
            if len(output_layers) == 1 and output_layers[0] == 'default':
                return x
    
            raise ValueError('output_layer is wrong.')

def resnet18(clip_len, output_layers=None, pretrained=True, pretrained2d=None, dilation_factor=1):
    """Constructs a ResNet-50 model.
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
    
    model = ResNet(clip_len, output_layers, pretrained)
    if(pretrained):
        model.Net2d.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet50(clip_len, output_layers=None, pretrained=True, pretrained2d=None, dilation_factor=1):
    """Constructs a ResNet-50 model.
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
    
    model = ResNet(output_layers, pretrained)
    if(pretrained):
        model.Net2d.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        checkpoint3d = torch.load('/home/mlcv/bdrhn9_ws/pytracking/pytracking/networks/resnet-50-kinetics.pth')
        state_dict3d = checkpoint3d['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
    
        for k, v in state_dict3d.items():
            new_state_dict[k[7:]]=v
        model.Net3d.load_state_dict(new_state_dict,strict=False)
    return model

def resnet50Trained(clip_len, output_layers=None, pretrained=True, pretrained2d=None, dilation_factor=1):
    """Constructs a ResNet-50 model.
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))
    
    model = ResNet(output_layers, pretrained)
    checkpoint = torch.load('/run/media/mlcv/DATA/pytracking/pytracking/networks/DiMPFMF_Con_Attn_50.pth.tar')
    if(pretrained):
        state_dict = checkpoint['net']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
    
        for k, v in state_dict.items():
            if k.split('.')[0] == 'feature_extractor':
                new_state_dict[k[18:]]=v
        model.load_state_dict(new_state_dict,strict=True)
    return model


if __name__ == '__main__':
    model = resnet18(output_layers=['layer2', 'layer3'], pretrained=True)
    # input_2d = torch.randn((2,3,224,224))
    input_3d = torch.randn((1,3,4,288,288))
    out_3d = model(input_3d)
    # out_3d = model.layers_3d(input_3d)
