# Visual Object Tracking by Using Ranking Loss and Spatial-Temporal Features (RankingAF and RankingSF)
The official implementation of the Visual Object Tracking by Using Ranking Loss and Spatial-Temporal Features (RankingAF and RankingSF). The article that summarizes this method is submitted to Machine Vision and Applications (MVAP). The article details will be given here after the review process.

### Trackers
These project introduces a novel two-stream deep neural network tracker for robust object tracking. In the proposed network, we use both spatial and temporal features and employ a novel loss function called ranking loss. The class confidence scores coming from the two-stream (spatial and temporal) networks are fused at the end for final decision. Using ranking loss in the proposed tracker enforces the networks to learn giving higher scores to the candidate regions that frame the target object better. As a result, the tracker returns more precise bounding boxes framing the target object, and the risk of tracking error accumulation and drifts are largely mitigated when the proposed network architecture is used with a simple yet effective model update rule.

#### The proposed two stream tracker
The architecture of the proposed two-stream network: We learn class-specific weights to fuse classifiers’ outputs of the two networks.
<img src="gfx/RankingOverall.png" width="750">

#### The differences between RankingT (published at ICCVW2019) and the last proposed methods (RankingAF, RankingSF) introduced in journal paper 
The preliminary version of this study has appeared in ICCVW2019. In the conference paper, we introduced RankingT method which only uses appearance information. The main differences between RankingT and the most recent method can be summarized as follows:
1) In the last version, we introduced novel dual-stream deep neural network architectures (RankingAF and Ranking SF) that use both spatial and temporal features. In contrast, RankingT method used only spatial information.
3) The last proposed method uses two novel fusion techniques to combine the scores coming from the spatial and temporal networks.
4) We also proposed a hard ranking mining technique to improve the performance of the ranking loss more in the last proposed method.
5) We provided more experiments on new datasets including TC-128, UAV123, NfS and DTB70 in the journal version.
6) We made a more detailed analysis of the recent related work on tracking (especially on tracking methods using spatio-temporal features).

## Results
#### The values of the online learned fusion weights for spatial and temporal networks for the zebrafish video frames

<img src="gfx/ChangeofWeights.png" width="500">

#### Visual Comparison 
Visual comparison of the our proposed method RankingSF with MDNet and ECO. The red, blue and green rectangles represent RankingSF, MDNet and ECO trackers, respectively.

<img src="gfx/RankingVisualComp.png" width="500">

#### State of the art comparison
We compared the proposed trackers, RankingAF and RankingSF (remember that RankingAF simply adds the scores of two networks directly whereas RankingSF learns network
specific weights as described earlier), on 6 different benchmark datasets.
#### Success and precision plots of the proposed methods and the state-of-the-art trackers on the OTB-2015 dataset.
<p float="left">
  <img src="Results/quality_plot_error_OPE_threshold_OTB100.png" width="300"/>
  <img src="Results/quality_plot_overlap_OPE_AUC_OTB100.png" width="300"/>
</p>

#### Success and precision plots of the proposed methods and the state-of-the-art trackers on the UAV123 dataset.

<p float="left">
  <img src="Results/precision_plot_uav.png" width="300"/>
  <img src="Results/success_plot_uav.png" width="300"/>
</p>

#### Comparison of the performances for different attributes on OTB-2015 dataset.

<p float="left">
  <img src="Results/abrupt_motion_overlap_OPE_AUC_OTB100.png" width="200"/>
  <img src="Results/deformation_overlap_OPE_AUC_OTB100.png" width="200"/>
  <img src="Results/out-of-plane_rotation_overlap_OPE_AUC_OTB100.png" width="200"/>
  <img src="Results/scale_variations_overlap_OPE_AUC_OTB100.png" width="200"/>
</p>

# RankingT: Visual object tracking by using Ranking Loss 
The official implementation of the ICCVW2019 paper [Visual object tracking by using Ranking Loss](http://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Cevikalp_Visual_Object_Tracking_by_Using_Ranking_Loss_ICCVW_2019_paper.pdf).

#### The proposed Ranking loss
<img src="gfx/loss_fcn.png" width="500">

#### Acknowledgements
This is a modified version of the python frameworks pytracking and py-MDNet based on PyTorch. We would like to thank the authors Martin Danelljan, Goutam Bhat, Hyeonseob Nam and Bohyung Han  for providing such a great frameworks.

## Citation
If you're using this code in a publication, please cite our paper.

```
@InProceedings{Cevikalp_2019_ICCV_Workshops,  
author = {Cevikalp, Hakan and Saribas, Hasan and Benligiray, Burak and Kahvecioglu, Sinem},  
title = {Visual Object Tracking by Using Ranking Loss},  
booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},  
month = {Oct},  
year = {2019}  
}
```
