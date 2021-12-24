import os
import scipy.io
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
        optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
        # optimizer = optim.Adam(param_list, lr = lr, weight_decay=w_decay)
    
    return optimizer


class MDNet_of(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet_of, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(2, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512*3*3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 1)) for _ in range(K)])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers, strict=False)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512*3*3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 1)) for _ in range(K)])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.reshape(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers, strict=False)
        
    # def load_model(self, model_path):
    #     states = torch.load(model_path)
    #     state_dict = states['model_state_dict']
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         if 'branches' in k:
    #             continue
    #         name = k # remove `module.`
    #         new_state_dict[name] = v
    #     self.load_state_dict(new_state_dict, strict=False)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])

class MDNetSF(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNetSF, self).__init__()
        self.K = K
        self.conv_rgb = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True)))]))                                                                          
        self.fc_rgb =  nn.Sequential(
                                nn.Linear(512 * 3 * 3, 512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(512, 512),
                                nn.ReLU(inplace=True)
                                )                                 
             
        self.branches_rgb = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 1)) for _ in range(K)])        
        
        self.conv_of = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True)))]))
                                                                             
        self.fc_of =  nn.Sequential(
                                nn.Linear(512 * 3 * 3, 512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(512, 512),
                                nn.ReLU(inplace=True)
                                )                                 
             
        self.branches_of = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 1)) for _ in range(K)]) 

        self.fusion = nn.Sequential(nn.Linear(2,1))     

        for m in self.fc_rgb.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.fc_of.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)                
        for m in self.branches_rgb.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.branches_of.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
                
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))        

        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches_rgb):
            append_params(self.params, module, 'fc6_{:d}'.format(k))
        for k, module in enumerate(self.branches_of):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self):
        for name, child in self.named_children():
#            print('name:', name,  'child:', child)
            if name in ['conv_rgb','conv_of', 'attention', 'conv_after_attention']:
                # print(name + 'has been frozen.')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + 'has not been frozen.')
                for param in child.parameters():
                    param.requires_grad = True

    def get_learnable_params(self):
#        self.params = self.state_dict()
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
#        self.params = self.state_dict()
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x_rgb, x_of, k=0, in_layer='conv1', out_layer='fc6'):     
        if in_layer == 'conv1':
            x_rgb = self.conv_rgb(x_rgb)
            x_rgb = x_rgb.reshape(x_rgb.size(0), -1)
            x_of = self.conv_of(x_of)
            x_of = x_of.reshape(x_of.size(0), -1)
        if out_layer == 'conv':
            return x_rgb, x_of
        x_rgb = self.fc_rgb(x_rgb)
        x_rgb = self.branches_rgb[k](x_rgb)
        x_of = self.fc_of(x_of)
        x_of = self.branches_of[k](x_of)   
        x = torch.cat((x_rgb, x_of), dim=1)
        x = self.fusion(x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        state_dict = states['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'branches' in k:
                continue
            name = k # remove `module.`
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict, strict=False)


    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.conv_rgb[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.conv_rgb[i][0].bias.data = torch.from_numpy(bias[:, 0])
            self.conv_of[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.conv_of[i][0].bias.data = torch.from_numpy(bias[:, 0])
            
class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss
    
class RankingLoss(nn.Module):
    def __init__(self):
        super(RankingLoss, self).__init__()
            
    def forward(self, pos_score, neg_score, rank_score, rank_ors, gains):
       
        torch_ones = Variable(torch.ones([len(neg_score), 1]).cuda(), requires_grad=False)
        torch_zeros = Variable(torch.zeros([len(neg_score), 1]).cuda(), requires_grad=False)
        loss_neg = torch.mean(torch.max(torch_zeros, torch_ones + neg_score))
   
        torch_ones = Variable(torch.ones([len(pos_score), 1]).cuda(), requires_grad=False)
        torch_zeros = Variable(torch.zeros([len(pos_score), 1]).cuda(), requires_grad=False)
        loss_pos = torch.mean(torch.max(torch_zeros, torch_ones - pos_score))
        loss_hinge = torch.add(loss_neg, loss_pos)
       
        
        #ranking loss
        rank_score = rank_score.view(int(rank_score.shape[0]/2),2)
        rank_score_diff =  rank_score[:,0] - rank_score[:,1]
        rank_score_diff = rank_score_diff.unsqueeze(dim=1)
        rank_ors = Variable(rank_ors.reshape(int(rank_ors.shape[0]/2),2))
        rank_ors_diff = rank_ors[:,0] - rank_ors[:,1]
        rank_ors_diff = rank_ors_diff.unsqueeze(dim=1)
        torch_zeros = Variable(torch.zeros([len(rank_score_diff), 1]).cuda(), requires_grad=False)
        rank_threshold = Variable(torch.zeros([len(rank_score_diff), 1]).fill_(0.5).cuda(), requires_grad=False)
        loss_ranking = torch.max(torch_zeros, rank_threshold-rank_score_diff)*rank_ors_diff/2
        loss_rank = torch.mean(loss_ranking)
        loss = torch.add(gains[0]*loss_hinge, gains[1]*loss_rank)
        #print(loss)
        return loss
    
    
class RankingLossSF(nn.Module):
    def __init__(self):
        super(RankingLossSF, self).__init__()
            
    def forward(self, pos_score, neg_score, rank_score, rank_ors, weights, gains):
       
        torch_ones = Variable(torch.ones([len(neg_score), 1]).cuda(), requires_grad=False)
        torch_zeros = Variable(torch.zeros([len(neg_score), 1]).cuda(), requires_grad=False)
        loss_neg = torch.mean(torch.max(torch_zeros, torch_ones + neg_score))
   
        torch_ones = Variable(torch.ones([len(pos_score), 1]).cuda(), requires_grad=False)
        torch_zeros = Variable(torch.zeros([len(pos_score), 1]).cuda(), requires_grad=False)
        loss_pos = torch.mean(torch.max(torch_zeros, torch_ones - pos_score))
        loss_hinge = torch.add(loss_neg, loss_pos)
      
        torch_zeros = Variable(torch.zeros([len(weights), 1]).cuda(), requires_grad=False)
        torch_ones = Variable(torch.ones([len(weights), 1]).cuda(), requires_grad=False)
        loss_w0 = torch.mean(torch.max(torch_zeros, -weights[0][0]))
        loss_w1 = torch.mean(torch.max(torch_zeros, -weights[0][1]))
        loss_aff = torch.mul((torch_ones - (weights[0][0] + weights[0][1])), (torch_ones - (weights[0][0] + weights[0][1])))
        
        #ranking loss
        rank_score = rank_score.view(int(rank_score.shape[0]/2),2)
        rank_score_diff =  rank_score[:,0] - rank_score[:,1]
        rank_score_diff = rank_score_diff.unsqueeze(dim=1)
        rank_ors = Variable(rank_ors.reshape(int(rank_ors.shape[0]/2),2))
        rank_ors_diff = rank_ors[:,0] - rank_ors[:,1]
        rank_ors_diff = rank_ors_diff.unsqueeze(dim=1)
        torch_zeros = Variable(torch.zeros([len(rank_score_diff), 1]).cuda(), requires_grad=False)
        rank_threshold = Variable(torch.zeros([len(rank_score_diff), 1]).fill_(0.5).cuda(), requires_grad=False)
        loss_ranking = torch.max(torch_zeros, rank_threshold-rank_score_diff)*rank_ors_diff/2
        loss_rank = torch.mean(loss_ranking)
        loss = torch.add(gains[0]*loss_hinge, gains[1]*loss_rank + loss_aff)
        #print(loss)
        return loss
    
    

class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


#class Precision():
#    def __call__(self, pos_score, neg_score):
#        scores = torch.cat((pos_score[:], neg_score[:]), 0)
#        topk = torch.topk(scores, pos_score.size(0))
#        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
#        return prec.item()

class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score, neg_score), 0)
        scores = scores.reshape(len(scores))
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        return prec.item()
    

