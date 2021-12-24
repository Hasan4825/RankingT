from . import BaseActor


class TratActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight
        
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        # iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'],
                                           train_labels=data['train_label'])

        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'].permute(1,0,2,3), data['test_anno'])
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test


        # iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        # iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])
        loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_loss_test

        # Total Loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf 

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        return loss, stats
  


