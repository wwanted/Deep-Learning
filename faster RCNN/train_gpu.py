import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
import tools_gpu
import util_gpu
import network_gpu
import time
import copy
from torchnet.meter import ConfusionMeter

PATH = 'D:/ReCT/img/'


def params_loss(pred_params, gt_params, gt_label):
    weight = torch.zeros_like(gt_params).cuda()
    mask = (gt_label > 0).view(-1, 1).expand_as(weight).cuda()
    weight[mask] = 1.
    loss = F.smooth_l1_loss(pred_params*weight, gt_params*weight, reduction='sum')
    loss = loss / ((gt_label >= 0).sum().float())
    return loss


class Trainer(nn.Module):
    def __init__(self, faster_rcnn, LR=0.001):
        super(Trainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.rpn_lamda = 3
        self.roi_lamda = 1
        self.anchor_target = tools_gpu.TargetAnchor()
        self.proposal_target = tools_gpu.TargetProposal()
        self.norm_mean = faster_rcnn.norm_mean
        self.norm_std = faster_rcnn.norm_std
        self.optimizer = torch.optim.Adam(self.faster_rcnn.parameters(), lr=LR)
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(1002)
        self.loss = [[], [], [], [], []]
        self.file1 = open('losses.txt', 'a')
        self.file2 = open('rpn_cm', 'a')
        self.file3 = open('roi_cm', 'a')

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, h, w = imgs.shape
        img_size = (h, w)
        features = self.faster_rcnn.extractor(imgs)
        rpn_params, rpn_scores, rois, roi_index, anchor = self.faster_rcnn.rpn(features, img_size, scale)
        bbox = bboxes[0]
        label = labels[0]
        rpn_params = rpn_params[0]
        rpn_score = rpn_scores[0]

        sample_roi, sample_params, sample_label = self.proposal_target(
            rois, bbox.detach().cpu().numpy(), label.detach().cpu().numpy(), self.norm_mean, self.norm_std)
        sample_roi_index = np.zeros((len(sample_roi),))
        roi_params, roi_scores = self.faster_rcnn.head(features, sample_roi, sample_roi_index)

        gt_rpn_params, gt_rpn_label = self.anchor_target(bbox.detach().cpu().numpy(), anchor, img_size)
        gt_rpn_params = torch.from_numpy(gt_rpn_params).float().cuda()
        gt_rpn_label = torch.from_numpy(gt_rpn_label).long().cuda()
        rpn_params_loss = self.rpn_lamda * params_loss(rpn_params, gt_rpn_params, gt_rpn_label)
        rpn_class_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]
        self.rpn_cm.add(_rpn_score.data, _gt_rpn_label.data)
        with open('rpn_cm.txt', 'a') as file1:
            file1.write(str(self.rpn_cm.value())+'\n\n')

        n_sample = roi_params.shape[0]
        roi_params = roi_params.view(n_sample, -1, 4)
        sample_label = torch.from_numpy(sample_label).long().cuda()
        sample_params = torch.from_numpy(sample_params).float().cuda()
        roi_params = roi_params[torch.arange(n_sample).long().cuda(), sample_label]

        roi_params_loss = self.roi_lamda * params_loss(roi_params.contiguous(), sample_params, sample_label)
        roi_class_loss = F.cross_entropy(roi_scores, sample_label)
        self.roi_cm.add(roi_scores.data, sample_label.data)
        with open('roi_cm.txt', 'a') as file2:
            file2.write(str(self.roi_cm.value())+'\n\n')

        losses = [rpn_params_loss, rpn_class_loss, roi_params_loss, roi_class_loss]
        losses = losses + [sum(losses)]

        return losses

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        for i in range(len(losses)):
            self.loss[i] = losses[i].cpu().item()
        loss_copy = copy.deepcopy(self.loss)
        with open('losses.txt', 'a') as file3:
            file3.write(str(loss_copy)+'\n\n')
        losses[4].backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.faster_rcnn.state_dict(), 'faster_rcnn_parameters.pkl')

    def load(self):
        self.faster_rcnn.load_state_dict(torch.load('faster_rcnn_parameters.pkl'))


def train():
    start_time = time.time()
    train_dataset = util_gpu.GetDataset(PATH)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    faster_rcnn = network_gpu.FasterRCNNVGG16()
    trainer = Trainer(faster_rcnn).cuda()
    trainer.load()

    for epoch in range(1):
        for i, (img, bbox, label, scale) in enumerate(dataloader):
            if len(bbox) == 0:
                continue
            scale = scale.item()
            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            trainer.train_step(img, bbox, label, scale)
            if i % 100 == 99:
                trainer.save()
            end_time = time.time()
            time_space = end_time - start_time
            start_time = end_time
            print('已训练第%d张图 ' % i, ' 时间消耗：%.2f秒 ' % time_space,
                  ' RPN Losses: %.4f, %.4f ' % (trainer.loss[0], trainer.loss[1]),
                  ' ROI Losses: %.4f, %.4f ' % (trainer.loss[2], trainer.loss[3]),
                  ' Total Losses: %.4f ' % trainer.loss[4])


if __name__ == '__main__':
    train()

















