import numpy as np
import torch
from torchvision.models import vgg16
from torch import nn
from torch.nn import functional as F
import tools_gpu
import time


def normal_init(model, mean, std, bias_mean, bias_std):
    model.weight.data.normal_(mean, std)
    model.bias.data.normal_(bias_mean, bias_std)


def basic_vgg16(pre=True):
    model = vgg16(pretrained=pre)
    features = list(model.features)[:30]
    classifier = list(model.classifier)
    del classifier[6]
    classifier = nn.Sequential(*classifier)
    for layer in features[:10]:
        for params in layer.parameters():
            params.requires_grad = False

    return nn.Sequential(*features), classifier


def roi_pooling(x, rois, size, scale):
    model = nn.AdaptiveMaxPool2d(size)
    _, _, h, w = x.shape
    output = []
    num = rois.shape[0]

    rois[:, 1:].mul_(scale).round_()
    rois[:, 1].clamp_(0, w)
    rois[:, 3].clamp_(0, w)
    rois[:, 2].clamp_(0, h)
    rois[:, 4].clamp_(0, h)
    rois = rois.long()

    for i in range(num):
        roi = rois[i]
        batch_index = roi[0]
        if roi[1] >= roi[3] or roi[2] >= roi[4]:
            clip_feature = model(x)
            clip_feature.fill_(0)
        else:
            clip = x.narrow(0, batch_index, 1)[:, :, roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
            clip_feature = model(clip)

        output.append(clip_feature)
    output = torch.cat(output, dim=0)

    return output


class RPN(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, feat_stride=16, proposal_params=dict()):
        super(RPN, self).__init__()
        self.feat_stride = feat_stride
        self.proposal = tools_gpu.Proposal(parent_model=self, **proposal_params)
        self.n_anchor = 9
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.score = nn.Conv2d(mid_channels, self.n_anchor*2, kernel_size=1, stride=1, padding=0)
        self.params = nn.Conv2d(mid_channels, self.n_anchor*4, kernel_size=1, stride=1, padding=0)
        normal_init(self.conv, 0, 0.01, 0, 0.01)
        normal_init(self.score, 0, 0.01, 0, 0.01)
        normal_init(self.params, 0, 0.01, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        anchor = tools_gpu.generate_anchor(self.feat_stride, h, w)
        mid_x = F.relu(self.conv(x))

        rpn_params = self.params(mid_x)
        rpn_params = rpn_params.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_score = self.score(mid_x)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_sm_score = F.softmax(rpn_score, dim=2)
        rpn_sm_score = rpn_sm_score[:, :, 1].contiguous().view(n, -1)

        rois = []
        roi_index = []
        for i in range(n):
            roi = self.proposal(rpn_params[i].cpu().data.numpy(), rpn_sm_score[i].cpu().data.numpy(),
                                anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_index.append(batch_index)

        rois = np.vstack(rois)
        roi_index = np.concatenate(roi_index, axis=0)

        return rpn_params, rpn_score, rois, roi_index, anchor


class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.params = nn.Linear(4096, n_class*4)
        self.score = nn.Linear(4096, n_class)
        self.n_class = n_class
        self.roi_size = roi_size
        self.scale = scale
        normal_init(self.params, 0, 0.001, 0, 0.001)
        normal_init(self.score, 0, 0.01, 0, 0.01)

    def forward(self, x, rois, roi_index):
        roi_index = torch.from_numpy(roi_index).float()
        rois = torch.from_numpy(rois).float()
        index_and_rois = torch.cat((roi_index[:, None], rois), dim=1).contiguous()

        clip_features = roi_pooling(x, index_and_rois, self.roi_size, self.scale)
        clip_features = clip_features.contiguous().view((clip_features.size(0), -1))
        fc7 = self.classifier(clip_features)
        roi_class_params = self.params(fc7)
        roi_scores = self.score(fc7)
        return roi_class_params, roi_scores


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, norm_mean=(0., 0., 0., 0.), norm_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.n_class = self.head.n_class
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.score_thresh = 0.01
        self.nms = 0.3

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        features = self.extractor(x)
        rpn_params, rpn_scores, rois, roi_index, anchor = self.rpn(features, img_size, scale)
        roi_class_params, roi_scores = self.head(features, rois, roi_index)
        return roi_class_params, roi_scores, rois, roi_index

    def predict(self, imgs, sizes):
        self.eval()
        bboxes = []
        labels = []
        scores = []
        for img, size in zip(imgs, sizes):
            img = torch.from_numpy(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_class_params, roi_scores, rois, _ = self(img, scale=scale)
            roi_class_params = roi_class_params.data
            roi_scores = roi_scores.data
            roi = torch.from_numpy(rois) / scale

            mean = torch.Tensor(self.norm_mean).cuda().repeat(self.n_class)[None]
            std = torch.Tensor(self.norm_std).cuda().repeat(self.n_class)[None]
            roi_class_params = (roi_class_params * std) + mean
            roi_class_params = roi_class_params.view(-1, self.n_class, 4)

            roi = roi.view(-1, 1, 4).expand_as(roi_class_params)
            class_bbox = tools_gpu.projectbbox(roi.detach().cpu().numpy().reshape((-1, 4)),
                                           roi_class_params.detach().cpu().numpy().reshape((-1, 4)))
            class_bbox = torch.from_numpy(class_bbox).view(-1, self.n_class*4)
            class_bbox[:, 0::2].clamp_(0, size[1])
            class_bbox[:, 1::2].clamp_(0, size[0])

            raw_prob = F.softmax(roi_scores, dim=1).detach().cpu().numpy()
            raw_bbox = class_bbox.detach().cpu().numpy()

            bbox, label, score = self.modify(raw_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(scale)

            self.train()
            return bboxes, labels, scores

    def modify(self, raw_bbox, raw_prob):
        bbox = []
        label = []
        score = []
        for i in range(1, self.n_class):
            bbox_i = raw_bbox.reshape((-1, self.n_class, 4))[:, i, :]
            prob_i = raw_prob[:, i]
            mask = prob_i > self.score_thresh
            prob_i = prob_i[mask]
            bbox_i = bbox_i[mask]

            order = prob_i.argsort()[::-1].astype(np.int32)
            bbox_i = bbox_i[order]
            keep = tools_gpu.nms_cpu(bbox_i, self.nms)

            bbox.append(bbox_i[keep])
            label.append((i-1)*np.ones((len(keep),)))
            score.append(prob_i[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score


class FasterRCNNVGG16(FasterRCNN):
    def __init__(self, n_fg_class=1001):
        extractor, classifier = basic_vgg16()
        rpn = RPN(512, 512, feat_stride=16,)
        head = VGG16RoIHead(n_class=n_fg_class+1, roi_size=7, scale=1/16, classifier=classifier)
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)

















































