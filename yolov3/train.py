import time
import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
from util import *
from darknet import Darknet
from torch.utils.data import Dataset, DataLoader
from skimage import io
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import imgaug.augmenters as iaa


def letterbox_image(img, bbox, inp_dim):
    # resize image with unchanged aspect ratio using padding
    img_w, img_h = img.shape[1], img.shape[0]
    h, w = inp_dim
    scale = min(w / img_w, h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    dx = (w - new_w) // 2
    dy = (h - new_h) // 2

    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[dy:dy + new_h, dx:dx + new_w, :] = resized_image

    bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale + dx
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale + dy

    return canvas, bbox


def bbox_iou_ssp(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    bl = np.maximum(bbox_a[:, np.newaxis, :2], bbox_b[:, :2])
    tr = np.minimum(bbox_a[:, np.newaxis, 2:], bbox_b[:, 2:])

    area = np.prod(tr - bl, axis=2) * (tr > bl).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    iou = area / (area_a[:, np.newaxis] + area_b - area + 1e-07)

    return iou


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    box_loss_scale = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]), 1),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors_ = anchors.copy()
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        wh_ = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    tx = true_boxes[b,t,0]*grid_shapes[l][1] - i
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    ty = true_boxes[b,t,1]*grid_shapes[l][0] -j
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')

                    scale = 2. - true_boxes[b,t,2]*true_boxes[b,t,3]
                    pw, ph = np.log(wh_[t]/anchors_[n])
                    box_loss_scale[l][b, j, i, k, 0] = scale

                    y_true[l][b, j, i, k, 0] = tx
                    y_true[l][b, j, i, k, 1] = ty
                    y_true[l][b, j, i, k, 2] = pw
                    y_true[l][b, j, i, k, 3] = ph
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    for l in range(num_layers):
        y_true[l] = y_true[l].reshape(-1, grid_shapes[l][0]*grid_shapes[l][1]*len(anchor_mask[l]), 5+num_classes)
        box_loss_scale[l] = box_loss_scale[l].reshape(-1, grid_shapes[l][0]*grid_shapes[l][1]*len(anchor_mask[l]), 1)

    y_true = np.concatenate(y_true, axis=1)
    box_loss_scale = np.concatenate(box_loss_scale, axis=1)
    return y_true, box_loss_scale


def cal_loss(y_predict, y_true, true_box, pred_box, box_loss_scale):
    n = y_predict.shape[0]
    object_mask = y_true[..., 4:5]
    pred_box = pred_box.data.numpy()
    ignore_mask = []

    for i in range(n):
        box_t = true_box[i, :, :4]
        validmask = box_t[:, 2] > 0
        box_t = box_t[validmask]
        if len(box_t) == 0:
            box_t = np.zeros((1, 4))
        box_p = pred_box[i, :, :4]
        iou = bbox_iou_ssp(box_p, box_t)
        best_iou = iou.max(-1)
        mask = best_iou < 0.5
        ignore_mask.append(mask)

    ignore_mask = np.vstack(ignore_mask)[..., None].astype(int)
    ignore_mask = torch.from_numpy(ignore_mask).float()
    box_loss_scale = torch.from_numpy(box_loss_scale).float()

    y_true_xy = torch.from_numpy(y_true[..., :2]).float()
    y_true_wh = torch.from_numpy(y_true[..., 2:4]).float()
    object_mask = torch.from_numpy(object_mask).float()
    y_true_class = torch.from_numpy(y_true[..., 5:]).float()

    xy_loss = F.mse_loss(y_predict[..., :2], y_true_xy, reduction='none') * object_mask * box_loss_scale
    wh_loss = F.mse_loss(y_predict[..., 2:4], y_true_wh, reduction='none') * object_mask * box_loss_scale
    xy_loss = 0.5 * xy_loss.sum() / n
    wh_loss = wh_loss.sum() / n

    object_loss_pos = object_mask * F.binary_cross_entropy(y_predict[..., 4:5], object_mask, reduction='none')
    object_loss_neg = (1-object_mask) * ignore_mask * F.binary_cross_entropy(y_predict[..., 4:5], object_mask, reduction='none')
    focal_mask = torch.pow(torch.abs(object_mask - y_predict[..., 4:5]), 2)
    object_loss = focal_mask * (object_loss_pos + object_loss_neg)
    object_loss = object_loss.sum() / n

    class_loss = object_mask * F.binary_cross_entropy(y_predict[..., 5:], y_true_class, reduction='none')
    class_loss = class_loss.sum() / n

    total_loss = xy_loss + wh_loss + object_loss + class_loss

    return [total_loss, xy_loss, wh_loss, object_loss, class_loss]


class Trainer(object):
    def __init__(self, lr=0.001, weight_decay=0.0001):
        self.LR = lr
        self.weight_decay = weight_decay
        self.confidence = 0.5
        self.nms_thesh = 0.4
        self.classes = load_classes('data/coco.names')
        self.num_classes = len(self.classes)

        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            self.model = Darknet('cfg/yolov3_coco.cfg').cuda()
        else:
            self.model = Darknet('cfg/yolov3_coco.cfg')
        self.model.load_weights('yolov3.weights')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.weight_decay)
        # for p in self.model.parameters():
            # print(p.requires_grad)

    def __call__(self, imgs, labels):
        precision_num = 0
        recall_num = 0
        class_right = 0
        precision_total = 0
        recall_total = 0

        true_box = labels.data.numpy()
        true_box_copy = true_box.copy()
        if self.CUDA:
            imgs = imgs.cuda()
        prediction, pred_box, input_size, anchors, y_pred = self.model(imgs, self.CUDA, True)
        y_true, box_loss_scale = preprocess_true_boxes(true_box, (input_size, input_size), anchors, self.num_classes)
        loss = cal_loss(prediction, y_true, true_box_copy, pred_box, box_loss_scale)

        self.optimizer.zero_grad()
        loss[0].backward()
        self.optimizer.step()

        loss_data = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(), loss[4].item()]
        with open('losses.txt', 'a') as file:
            file.write(str(loss_data) + '\n')

        y_pred = write_results(y_pred, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        if type(y_pred) == int:
            y_pred = np.zeros((1, 8))
            y_pred[:, 0] = -1
        else:
            y_pred = y_pred.cpu().data.numpy()

        for i in range(len(true_box_copy)):
            index = np.where(y_pred[:, 0] == i)
            box_p = y_pred[index]
            box_t = true_box_copy[i]
            validmask = box_t[:, 2] > 0
            box_t = box_t[validmask]

            if len(box_t) == 0:
                precision_total += len(box_p)
                continue
            if len(box_p) == 0:
                recall_total += len(box_t)
                continue

            box_p[:, 1:5] = np.clip(box_p[:, 1:5], 0, input_size)

            iou = bbox_iou_ssp(box_p[:, 1:5], box_t[:, :4])
            index1 = np.where(iou >= 0.5)[0]
            index2 = np.where(iou >= 0.5)[1]

            iou_p = iou[np.unique(index1)]
            arg_index = np.argmax(iou_p, axis=1)
            box_p = box_p[np.unique(index1)]
            box_t = box_t[arg_index]
            class_right += (box_p[:, -1] == box_t[:, -1]).sum()

            precision_num += len(np.unique(index1))
            precision_total += iou.shape[0]
            recall_num += len(np.unique(index2))
            recall_total += iou.shape[1]

        if precision_total == 0:
            precision_total = 1
        if recall_total == 0:
            recall_total = 1

        return loss_data, precision_num, precision_total, recall_num, recall_total, class_right

    def predict(self, imgs, labels):
        precision_num = 0
        recall_num = 0
        class_right = 0
        precision_total = 0
        recall_total = 0

        true_box = labels.data.numpy()
        true_box_copy = true_box.copy()
        if self.CUDA:
            imgs = imgs.cuda()
        prediction, pred_box, input_size, anchors, y_pred = self.model(imgs, self.CUDA, True)
        y_true, box_loss_scale = preprocess_true_boxes(true_box, (input_size, input_size), anchors, self.num_classes)
        loss = cal_loss(prediction, y_true, true_box_copy, pred_box, box_loss_scale)

        loss_data = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(), loss[4].item()]

        y_pred = write_results(y_pred, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        if type(y_pred) == int:
            y_pred = np.zeros((1, 8))
            y_pred[:, 0] = -1
        else:
            y_pred = y_pred.cpu().data.numpy()

        for i in range(len(true_box_copy)):
            index = np.where(y_pred[:, 0] == i)
            box_p = y_pred[index]
            box_t = true_box_copy[i]
            validmask = box_t[:, 2] > 0
            box_t = box_t[validmask]

            if len(box_t) == 0:
                precision_total += len(box_p)
                continue
            if len(box_p) == 0:
                recall_total += len(box_t)
                continue

            box_p[:, 1:5] = np.clip(box_p[:, 1:5], 0, input_size)

            iou = bbox_iou_ssp(box_p[:, 1:5], box_t[:, :4])
            index1 = np.where(iou >= 0.5)[0]
            index2 = np.where(iou >= 0.5)[1]

            iou_p = iou[np.unique(index1)]
            arg_index = np.argmax(iou_p, axis=1)
            box_p = box_p[np.unique(index1)]
            box_t = box_t[arg_index]
            class_right += (box_p[:, -1] == box_t[:, -1]).sum()

            precision_num += len(np.unique(index1))
            precision_total += iou.shape[0]
            recall_num += len(np.unique(index2))
            recall_total += iou.shape[1]

        if precision_total == 0:
            precision_total = 1
        if recall_total == 0:
            recall_total = 1

        return loss_data, precision_num, precision_total, recall_num, recall_total, class_right

    def predict_no_label(self, imgs):
        if self.CUDA:
            imgs = imgs.cuda()
        with torch.no_grad():
            prediction = self.model(imgs, self.CUDA, Train=False)
        prediction = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        return prediction

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))


class GetDataset(Dataset):
    def __init__(self, path, train=True, size=320):
        self.size = size
        self.train = train
        if train:
            self.imgpath = glob.glob(path + '/*.jpg')
            self.labels = np.load('labels.npy')
        else:
            self.imgpath = glob.glob(path + '/*.jpg')
            self.labels = np.load('test_labels.npy')

    def __getitem__(self, i):
        img = io.imread(self.imgpath[i])
        h, w, _ = img.shape
        bbox = self.labels[i].copy()

        bbox[:, 0] = bbox[:, 0] * w
        bbox[:, 1] = bbox[:, 1] * h
        bbox[:, 2] = bbox[:, 2] * w
        bbox[:, 3] = bbox[:, 3] * h

        new_box = bbox.copy()
        new_box[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
        new_box[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        new_box[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
        new_box[:, 3] = bbox[:, 1] + bbox[:, 3] / 2

        img, bbox = letterbox_image(img, new_box, (self.size, self.size))

        '''
        if self.train:
            aug_box = [BoundingBox(x[0], x[1], x[2], x[3]) for x in bbox]
            bbs = BoundingBoxesOnImage(aug_box, shape=img.shape)
            seq = iaa.Sequential([iaa.Fliplr(0.2), iaa.Flipud(0.2), iaa.Multiply((0.8, 1.2), per_channel=0.2),
                                  iaa.ContrastNormalization((0.8, 1.2), per_channel=0.2), iaa.SomeOf((0, None), [
                    iaa.CropAndPad(percent=(-0.1, 0.0)), iaa.GaussianBlur(sigma=(0.0, 0.5)),
                    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.95, 1.1),
                               rotate=(-5, 5))]), iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255), per_channel=0.2)],
                                 random_order=True)
            image_aug, bbs_aug = seq(image=img.astype(np.float32), bounding_boxes=bbs)
            for i in range(len(bbs_aug.bounding_boxes)):
                after = bbs_aug.bounding_boxes[i]
                bbox[i, :4] = after.x1, after.y1, after.x2, after.y2
            img = image_aug
        '''

        bbox[:, :4] = np.clip(bbox[:, :4], 0, self.size - 1)
        box_w = bbox[:, 2] - bbox[:, 0]
        box_h = bbox[:, 3] - bbox[:, 1]
        mask1 = box_w > 3
        mask2 = box_h > 3
        mask = mask1 * mask2
        bbox = bbox[mask]

        img = np.transpose(img, (2, 0, 1)) / 255

        box_data = np.zeros((50, 5))
        if len(bbox) > 50:
            np.random.shuffle(bbox)
            bbox = bbox[:50]

        box_data[:len(bbox)] = bbox

        return torch.from_numpy(img).float(), box_data

    def __len__(self):
        return len(self.imgpath)

    def __len__(self):
        return len(self.imgpath)


if __name__ == '__main__':

    start_time = time.time()
    train_dataset = GetDataset('D:/VOC/imgs')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    trainer = Trainer()
    trainer.load('voc_80.pkl')

    for epoch in range(100):
        pre = 0
        pre_all = 0
        rec = 0
        rec_all = 0
        class_r = 0

        for i, (imgs, labels) in enumerate(train_dataloader):
            loss, precision_num, precision_total, recall_num, recall_total, class_right = trainer(imgs, labels)
            pre += precision_num
            pre_all += precision_total
            rec += recall_num
            rec_all += recall_total
            class_r += class_right

            if i % 1 == 0:
                with open('voc.txt', 'a') as file:
                    file.write(str([epoch, i]) + ' Loss:{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                        *loss) + ' bbox_precision:%.4f' % (precision_num / precision_total) +
                               ' bbox_recall:%.4f' % (recall_num / recall_total) + ' class_precision:%.4f ' % (
                                           class_right / precision_total) +
                               ' class_recall:%.4f' % (class_right / recall_total) + ' Total_num:' + str(
                        [pre, pre_all, rec, rec_all, class_r]) + ' Total_parm:{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                        pre / pre_all, rec / rec_all, class_r / pre_all, class_r / rec_all) + '\n')
                end_time = time.time()
                time_space = end_time - start_time
                start_time = end_time
                print('Epoch:%d ' % epoch, ' Batch:%d ' % i, ' time：%.2f秒 ' % time_space,
                      'Loss:{:.4f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(*loss),
                      'precision:%.2f' % (precision_num / precision_total),
                      'recall:%.2f' % (recall_num / recall_total), 'class_pre:%.2f' % (class_right / precision_total),
                      'class_rec:%.2f' % (class_right / recall_total))

        if epoch % 10 == 0:
            trainer.save('voc_%d.pkl' % epoch)
        if epoch % 10 == 9:
            trainer.LR *= 0.8

