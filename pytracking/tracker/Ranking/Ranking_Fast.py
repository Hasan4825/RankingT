from pytracking.tracker.base import BaseTracker
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import math
import time
import os, sys, yaml, json
from PIL import Image
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
from pytracking.tracker.Ranking.bbreg import BBRegressor
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation
from pytracking.tracker.Ranking.data_prov import RegionExtractor
from pytracking.tracker.Ranking.sample_generator import *
from matplotlib import pyplot as plt
class Ranking(BaseTracker):

    multiobj_mode = 'parallel'
    
    def extract_regions_align(self, image, bbox, out_layer='conv3'):
        self.model.eval()
        bbox = bbox.copy()
        # image = np.asarray(image)
        # image = torchvision.transforms.ToTensor()(image).unsqueeze(0).cuda()
        # bbox[:,2:] += bbox[:,:2] + 2*self.params.padding*bbox[:,2:]/self.params.img_size 
        bbox[:,2:] += bbox[:,:2]
        bb_torch = torch.tensor(bbox).cuda()
        regions = torchvision.ops.roi_align(image, [bb_torch], self.params.img_size)
        with torch.no_grad():
            feats = self.model(regions, out_layer=out_layer).detach()
            torch.cuda.empty_cache()
        return feats
        
    
    def forward_samples(self, image, samples, out_layer='conv3'):
        self.model.eval()
        extractor = RegionExtractor(image, samples, self.params)
        for i, regions in enumerate(extractor):
            if self.params.use_gpu:
                regions = regions.cuda()
            with torch.no_grad():
                feat = self.model(regions, out_layer=out_layer)
            if i==0:
                feats = feat.detach().clone()
            else:
                feats = torch.cat((feats, feat.detach().clone()), 0)
        return feats
    
    def train(self, optimizer, pos_feats, neg_feats, rank_feats, rank_ors, maxiter, in_layer='fc4'):
        self.model.train()
        gains = self.params.gains
        batch_pos = self.params.batch_pos
        batch_neg = self.params.batch_neg
        batch_rank = self.params.batch_rank
        batch_test = self.params.batch_test
        batch_neg_cand = max(self.params.batch_neg_cand, batch_neg)
        # batch_rank_cand = max(self.params.batch_rank_cand, batch_rank)
    
        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))
        rank_idx = np.asarray(range(rank_feats.size(0)))
        while(len(pos_idx) < batch_pos * maxiter):
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while(len(neg_idx) < batch_neg_cand * maxiter):
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        while(len(rank_idx) < batch_rank*maxiter):
            rank_idx = np.concatenate([rank_idx, np.asarray(range(rank_feats.size(0)))])
    
        rank_ors = torch.cuda.FloatTensor(rank_ors)  
        pos_pointer = 0
        neg_pointer = 0
        rank_pointer = 0
        for i in range(maxiter):
    
            # select pos idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next
    
            # select neg idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next
            
             # select rank idx
            rank_next = rank_pointer+batch_rank
            rank_cur_idx = rank_idx[rank_pointer:rank_next]
            rank_cur_idx = rank_feats.new(rank_cur_idx).long()
            rank_pointer = rank_next
    
            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_neg_feats = neg_feats[neg_cur_idx]
            batch_rank_feats = rank_feats[rank_cur_idx]
            batch_rank_ors = rank_ors[rank_cur_idx]
           # print(iter)
            
    
     # hard negative mining
            if batch_neg_cand > batch_neg:
                self.model.eval()
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    with torch.no_grad():
                        score = self.model(batch_neg_feats[start:end], in_layer=in_layer)
                    if start==0:
                        neg_cand_score = score.detach()[:, 0].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 0].clone()), 0)
    
                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats[top_idx]
                self.model.train()
            if maxiter > 20 and i == 0: 
                Hard_rank = False
            else :
                Hard_rank = True
            if self.params.hard_rank:       
                # model.eval()
                rank_ors_up = batch_rank_ors[0::2]
                rank_ors_down = batch_rank_ors[1::2]
                rank_feats_up = batch_rank_feats[0::2,:]
                rank_feats_down = batch_rank_feats[1::2,:]
                rank_up_scores = self.model(rank_feats_up, in_layer=in_layer)
                rank_down_scores = self.model(rank_feats_down, in_layer=in_layer)
                rank_up_scores, ind_up = torch.sort(rank_up_scores, 0, descending=False)
                rank_down_scores, ind_down = torch.sort(rank_down_scores, 0, descending=True)
                rank_feats_up = rank_feats_up[ind_up].squeeze()
                rank_feats_down = rank_feats_down[ind_down].squeeze()
                batch_rank_feats[0::2,:] = rank_feats_up
                batch_rank_feats[1::2,:] = rank_feats_down
    #            rank_examples_up = rank_examples_up[ind_up.cpu().numpy(),:]
    #            rank_examples_down = rank_examples_down[ind_down.cpu().numpy(),:]
                rank_ors_up = rank_ors_up[ind_up]
                rank_ors_down = rank_ors_down[ind_down]
                batch_rank_ors[0::2] = rank_ors_up.squeeze()
                batch_rank_ors[1::2] = rank_ors_down.squeeze()
                # model.train()
    
            # forward
            pos_score = self.model(batch_pos_feats, in_layer=in_layer)
            neg_score = self.model(batch_neg_feats, in_layer=in_layer)
            rank_score = self.model(batch_rank_feats, in_layer=in_layer)
            # optimize
            loss = self.criterion(pos_score, neg_score, rank_score, batch_rank_ors, gains)
            # print('losss:', loss)
            self.model.zero_grad()
            loss.backward()
            # if 'grad_clip' in self.params:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_clip)
            optimizer.step()

    def Sample_select(self, image, target_bbox):
        pos_examples, pos_ors = SampleGenerator('gaussian', image.size, self.params.trans_pos, self.params.scale_pos)(
             target_bbox, self.params.n_pos_init, self.params.overlap_pos_init)
        neg_examples1, neg_ors1 = SampleGenerator('uniform', image.size, self.params.trans_neg_init, self.params.scale_neg_init)(
                            target_bbox, int(self.params.n_neg_init * 0.5), self.params.overlap_neg_init)
        neg_examples2, neg_ors2 = SampleGenerator('whole', image.size)(
                            target_bbox, int(self.params.n_neg_init * 0.5), self.params.overlap_neg_init)         
        neg_examples =  np.concatenate([neg_examples1, neg_examples2])
        neg_examples = np.random.permutation(neg_examples)
        
        rank_examples_up, rank_ors_up = SampleGenerator('gaussian', image.size, self.params.trans_rank_up, self.params.scale_rank_up)(
                 target_bbox, int(self.params.n_rank_init * 0.5), self.params.overlap_rank_init_up)
        rank_examples_down, rank_ors_down = SampleGenerator('gaussian', image.size, self.params.trans_rank_down, self.params.scale_rank_down)(
                 target_bbox, int(self.params.n_rank_init * 0.5), self.params.overlap_rank_init_down)   
        rank_examples = np.zeros((int(2*min(len(rank_examples_up),len(rank_examples_down))),4), dtype='float32')
        rank_ors = np.zeros((len(rank_examples),), dtype='float32')
        itr=0
        while itr<(len(rank_examples)/2):
            rank_examples[2*itr]=rank_examples_up[itr,:]
            rank_examples[2*itr+1]=rank_examples_down[itr,:]
            rank_ors[2*itr]=rank_ors_up[itr]
            rank_ors[2*itr+1]=rank_ors_down[itr]
            itr +=1
            
        bbreg_examples, bbreg_ors = SampleGenerator('uniform', image.size, self.params.trans_bbreg, self.params.scale_bbreg, self.params.aspect_bbreg)(
                        target_bbox, self.params.n_bbreg, self.params.overlap_bbreg)
        return  pos_examples, neg_examples, rank_examples, rank_ors, bbreg_examples

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        self.model = self.params.model
        if not self.params.has('device'):
            self.model = self.model.cuda()
        self.model.set_learnable_params(self.params.ft_layers)
        
        self.criterion = self.params.criterion
        self.init_optimizer = self.params.init_optimizer
        self.update_optimizer = self.params.update_optimizer
        self.image_size = image.size
        self.target_bbox = np.array(info['init_bbox'], dtype='float32')
        # self.result = np.zeros((len(img_list), 4))
        # self.result_bb = np.zeros((len(img_list), 4))
        # self.result[0] = target_bbox
        # self.result_bb[0] = target_bbox
        # The DiMP network
        # Time initialization
        tic = time.time()

        # Convert image
        # image = Image.fromarray(image)
        
        pos_examples, neg_examples, rank_examples, rank_ors, bbreg_examples = self.Sample_select(image, target_bbox=self.target_bbox)
 
        # for y in range(30):
        #     dpi = 80.0
        #     figsize = (image.size[0] / dpi, image.size[1] / dpi)
    
        #     fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     im = ax.imshow(image, aspect='auto')
    
        #     # if gt is not None:
        #         # gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
        #         #                         linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
        #         # ax.add_patch(gt_rect)
    
        #     rect = plt.Rectangle(tuple(rank_examples[y, :2]), rank_examples[y, 2], rank_examples[y, 3],
        #                         linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        #     ax.add_patch(rect)
        #     print("rank_ors:", rank_ors[y])
        #     plt.show()
        #     # plt.draw()
        # Extract Featuresn:
        # t1 = time.time()
        # feats = self.extract_regions_align(image, self.target_bbox)
        # print('Time:', time.time() - t1)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                    ),
            ])
        image = self.transform(image).unsqueeze(0).cuda()
        pos_feats = self.extract_regions_align(image, pos_examples)
        neg_feats = self.extract_regions_align(image, neg_examples)
        rank_feats = self.extract_regions_align(image, rank_examples)



        # im = numpy_to_torch(image)
        # İnitial training
        # t1 = time.time()

        self.train(self.init_optimizer, pos_feats, neg_feats, rank_feats, rank_ors, self.params.maxiter_init)
        # print('Time:', time.time() - t1)

        self.bbreg_feats = self.extract_regions_align(image, bbreg_examples)
        self.bbreg = BBRegressor(self.image_size)
        self.bbreg.train(self.bbreg_feats, bbreg_examples, self.target_bbox)
   
        
       # Init sample generators for update
        self.sample_generator = SampleGenerator('gaussian', self.image_size, self.params.trans, self.params.scale)
        self.pos_generator = SampleGenerator('gaussian', self.image_size, self.params.trans_pos, self.params.scale_pos)
        self.neg_generator = SampleGenerator('uniform', self.image_size, self.params.trans_neg, self.params.scale_neg)
        self.rank_generator_up = SampleGenerator('gauussian', self.image_size, self.params.trans_rank_up, self.params.scale_rank_up)
        self.rank_generator_down = SampleGenerator('gauussian', self.image_size, self.params.trans_rank_down, self.params.scale_rank_down)
        # Init pos/neg features for update
        neg_examples, _ = self.neg_generator(self.target_bbox, self.params.n_neg_update, self.params.overlap_neg_init)
        neg_feats = self.extract_regions_align(image, neg_examples)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]
        self.rank_feats_all = [rank_feats]
        rank_ors = torch.from_numpy(rank_ors).float().cuda()
        self.rank_ors_all = [rank_ors]
        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        image = self.transform(image).unsqueeze(0).cuda()

        samples = self.sample_generator(self.target_bbox, self.params.n_samples)
        sample_scores = self.extract_regions_align(image, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 0].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
            
        success = target_score > 0
        if success:
            self.sample_generator.set_trans(self.params.trans)
        else:
            self.sample_generator.expand_trans(self.params.trans_limit)
            
        # Convert image
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats = self.extract_regions_align(image, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
       
     # Data collect
        if success:
            pos_examples, _ = self.pos_generator(target_bbox, self.params.n_pos_update, self.params.overlap_pos_update)
            if len(pos_examples)<1 :
                pos_examples = target_bbox.reshape(1,4)
            pos_feats = self.extract_regions_align(image, pos_examples)
            self.pos_feats_all.append(pos_feats)
            if len(self.pos_feats_all) > self.params.n_frames_long:
                del self.pos_feats_all[0]
                del self.rank_feats_all[0]
                del self.rank_ors_all[0]

            neg_examples, _ = self.neg_generator(target_bbox, self.params.n_neg_update, self.params.overlap_neg_update)
            neg_feats = self.extract_regions_align(image, neg_examples)
            self.neg_feats_all.append(neg_feats)     
            # pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            rank_examples_up, rank_ors_up = self.pos_generator(target_bbox, int(self.params.n_rank_update * 0.5), self.params.overlap_rank_update_up)
            rank_examples_down, rank_ors_down = self.neg_generator(target_bbox, int(self.params.n_rank_update * 0.5), self.params.overlap_rank_update_down)
            rank_examples = np.zeros((int(2*min(len(rank_examples_up),len(rank_examples_down))),4), dtype='float32')
            rank_ors = np.zeros((len(rank_examples),), dtype='float32')
            itr=0
            while itr<(len(rank_examples)/2):
                rank_examples[2*itr]=rank_examples_up[itr,:]
                rank_examples[2*itr+1]=rank_examples_down[itr,:]
                rank_ors[2*itr]=rank_ors_up[itr]
                rank_ors[2*itr+1]=rank_ors_down[itr]
                itr +=1
            rank_ors = torch.from_numpy(rank_ors).float().cuda()
            self.rank_ors_all.append(rank_ors)
            rank_feats = self.extract_regions_align(image, rank_examples)
            self.rank_feats_all.append(rank_feats)

            if len(self.neg_feats_all) > self.params.n_frames_short:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(self.params.n_frames_short, len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            rank_data = torch.cat(self.rank_feats_all[-nframes:], 0)
            rank_ors_data = torch.cat(self.rank_ors_all[-nframes:], 0)
            self.train(self.update_optimizer, pos_data, neg_data, rank_data, rank_ors_data, self.params.maxiter_update)

        # Long term update
        elif (self.frame_num-1) % self.params.long_interval == 0:
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            rank_data = torch.cat(self.rank_feats_all, 0)
            rank_ors_data = torch.cat(self.rank_ors_all, 0)
            self.train(self.update_optimizer, pos_data, neg_data, rank_data, rank_ors_data, self.params.maxiter_update)

      
      # Compute output bounding box
        new_state = bbreg_bbox
        self.target_bbox = target_bbox

        if success is False:
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state}
        return out

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')