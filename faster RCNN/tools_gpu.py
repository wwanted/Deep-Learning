import numpy as np


def projectbbox(bbox, params):
    if bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=params.dtype)

    xmin, ymin, xmax, ymax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    h = ymax - ymin
    w = xmax - xmin
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    t_x, t_y, t_h, t_w = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    n_x = t_x * w + x
    n_y = t_y * h + y
    n_h = np.exp(t_h) * h
    n_w = np.exp(t_w) * w

    n_xmin = n_x - n_w / 2
    n_xmax = n_x + n_w / 2
    n_ymin = n_y - n_h / 2
    n_ymax = n_y + n_h / 2

    n_bbox = np.stack((n_xmin, n_ymin, n_xmax, n_ymax), axis=1)

    return n_bbox


def bbox_shift(pbox, gbox):
    pw = pbox[:, 2] - pbox[:, 0]
    ph = pbox[:, 3] - pbox[:, 1]
    px = pbox[:, 0] + pw / 2
    py = pbox[:, 1] + ph / 2

    gw = gbox[:, 2] - gbox[:, 0]
    gh = gbox[:, 3] - gbox[:, 1]
    gx = gbox[:, 0] + gw / 2
    gy = gbox[:, 1] + gh / 2

    ph = np.maximum(ph, 1)
    pw = np.maximum(pw, 1)

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = np.log(gw / pw)
    th = np.log(gh / ph)

    params = np.stack((tx, ty, th, tw), axis=1)

    return params


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    bl = np.maximum(bbox_a[:, np.newaxis, :2], bbox_b[:, :2])
    tr = np.minimum(bbox_a[:, np.newaxis, 2:], bbox_b[:, 2:])

    area = np.prod(tr - bl, axis=2) * (tr > bl).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    iou = area / (area_a[:, np.newaxis] + area_b - area)

    return iou


def anchor_base(base=16, scale=[1, 3, 8], ratio=[0.6, 1, 1.4]):
    bx = base / 2
    by = base / 2
    scale = np.array(scale)
    ratio = np.array(ratio)

    width = (scale * base)[:, None] * np.sqrt(ratio)
    height = (width / ratio).ravel('F')

    xmin = bx - width.ravel('F') / 2
    xmax = bx + width.ravel('F') / 2
    ymin = by - height / 2
    ymax = by + height / 2

    anchor = np.stack((xmin, ymin, xmax, ymax), axis=1)

    return anchor


def generate_anchor(feat_stride, height, width):
    base = anchor_base()
    x = np.arange(0, feat_stride * width, feat_stride)
    y = np.arange(0, feat_stride * height, feat_stride)

    x, y = np.meshgrid(x, y)
    shift = np.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)

    a = base.shape[0]
    k = shift.shape[0]
    all_anchor = shift[:, None, :] + base
    all_anchor = all_anchor.reshape((a * k, 4)).astype(np.float32)

    return all_anchor


def get_inside_bbox(anchor, h, w):
    index = np.where((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) &
                     (anchor[:, 2] <= w) & (anchor[:, 3] <= h))[0]
    return index


def unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class TargetAnchor(object):
    def __init__(self, sample=256, pos_iou=0.7, neg_iou=0.3, pos_ratio=0.5):
        self.sample = sample
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        h, w = img_size
        n = anchor.shape[0]
        inside_index = get_inside_bbox(anchor, h, w)
        anchor = anchor[inside_index]

        argmax, label = self._label(inside_index, anchor, bbox)
        params = bbox_shift(anchor, bbox[argmax])
        label = unmap(label, n, inside_index, fill=-1)
        params = unmap(params, n, inside_index, fill=0)

        return params, label

    def _label(self, index, anchor, bbox):
        label = np.empty((len(index),), dtype=np.int32)
        label.fill(-1)

        argmax, max_ious, gt_argmax = self._cal_iou(anchor, bbox)
        label[max_ious < self.neg_iou] = 0
        label[gt_argmax] = 1
        label[max_ious >= self.pos_iou] = 1

        n_pos = int(self.pos_ratio * self.sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = self.sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax, label

    def _cal_iou(self, anchor, bbox):
        iou = bbox_iou(anchor, bbox)
        argmax = iou.argmax(axis=1)
        max_ious = iou[np.arange(iou.shape[0]), argmax]
        gt_argmax = iou.argmax(axis=0)
        gt_maxious = iou[gt_argmax, np.arange(iou.shape[1])]

        gt_argmax1 = np.array([], dtype=np.int32)
        for i in range(iou.shape[1]):
            row_index = np.where(iou[:, i] == gt_maxious[i])[0]
            gt_argmax1 = np.append(gt_argmax1, row_index)

        gt_argmax = np.sort(np.unique(gt_argmax1))
        return argmax, max_ious, gt_argmax


class Proposal(object):
    def __init__(self, parent_model, nms=0.7, train_pre=12000, train_post=6000,
                 test_pre=6000, test_post=300, min_size=4):
        self.parent_model = parent_model
        self.nms = nms
        self.train_pre = train_pre
        self.train_post = train_post
        self.test_pre = test_pre
        self.test_post = test_post
        self.min_size = min_size

    def __call__(self, params, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            pre_nms = self.train_pre
            post_nms = self.train_post
        else:
            pre_nms = self.test_pre
            post_nms = self.test_post

        roi = projectbbox(anchor, params)
        roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, img_size[1])
        roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, img_size[0])

        min_size = self.min_size * scale
        w = roi[:, 2] - roi[:, 0]
        h = roi[:, 3] - roi[:, 1]
        index = np.where((w >= min_size) & (h >= min_size))[0]
        roi = roi[index, :]
        score = score[index]

        order = score.ravel().argsort()[::-1]
        if pre_nms > 0:
            order = order[:pre_nms]
        roi = roi[order, :]
        keep = nms_cpu(roi, self.nms)

        if post_nms > 0:
            keep = keep[:post_nms]
        roi = roi[keep]
        return roi


def nms_cpu(bbox, threshhold):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    index = np.arange(len(bbox), dtype=np.int32)
    keep = np.array([], dtype=np.int32)

    while len(index) > 0:
        keep = np.append(keep, index[0])
        bbox_a = bbox[index[0]][None, ]
        bbox_b = bbox[index[1:], :]
        iou = bbox_iou(bbox_a, bbox_b)
        hold = np.where(iou < threshhold)[1]+1
        index = index[hold]

    return keep


class TargetProposal(object):
    def __init__(self, sample=128, pos_ratio=0.5, pos_iou=0.5,
                 neg_iou_high=0.5, neg_iou_low=0.0):
        self.sample = sample
        self.pos_ratio = pos_ratio
        self.pos_iou = pos_iou
        self.neg_iou_high = neg_iou_high
        self.neg_iou_low = neg_iou_low

    def __call__(self, roi, bbox, label, mean=(0., 0., 0., 0.), std=(0.1, 0.1, 0.2, 0.2)):

        pos_num = np.round(self.pos_ratio * self.sample)
        roi = np.vstack((roi, bbox))

        iou = bbox_iou(roi, bbox)
        argmax = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        gt_label = label[argmax]+1
        pos_index = np.where(max_iou >= self.pos_iou)[0]
        neg_index = np.where((max_iou < self.neg_iou_high) & (max_iou > self.neg_iou_low))[0]

        pos_num = int(min(pos_num, len(pos_index)))
        if pos_num > 0:
            pos_index = np.random.choice(pos_index, size=pos_num, replace=False)

        neg_num = int(min((self.sample - pos_num), len(neg_index)))
        if neg_num > 0:
            neg_index = np.random.choice(neg_index, size=neg_num, replace=False)

        keep = np.append(pos_index, neg_index)
        sample_roi = roi[keep]
        sample_label = gt_label[keep]
        sample_label[pos_num:] = 0

        sample_params = bbox_shift(sample_roi, bbox[argmax[keep]])
        sample_params = (sample_params - np.array(mean)) / np.array(std)

        return sample_roi, sample_params, sample_label


























































