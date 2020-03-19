import numpy as np
import glob
from skimage import io


def letterbox_image(img, bbox, inp_dim):
    ori_height, ori_width = img.shape[:2]

    bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * inp_dim[0]

    bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * inp_dim[1]

    return bbox


def eval_truth(path):
    imgpath = glob.glob(path + '/*.jpg')
    labels = np.load('test_labels.npy')
    eval_labels = []
    for i in range(20):
        class_labels = []
        for j in range(len(labels)):
            img = io.imread(imgpath[j])
            bbox = labels[j].copy()

            bbox = letterbox_image(img, bbox, (640, 640))
            ind = np.zeros((len(bbox), 1))
            det = np.zeros((len(bbox), 1))
            ind.fill(j)
            bbox = np.concatenate((ind, bbox, det), 1)
            mask = bbox[:, 5] == i
            bbox = bbox[mask]
            class_labels.append(bbox)
        print(i)
        eval_labels.append(class_labels)
    return eval_labels


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_voc(pred, eval_label, class_index, ovthresh=0.5, use_07_metric=False):
    labels = eval_label[class_index]
    npos = 0
    for x in labels:
        npos += len(x)

    mask = (pred[:, -1] == class_index)
    confidence = pred[mask, 5]
    bbox = pred[mask, 1:5]
    img_ids = pred[mask, 0]

    sorted_ind = np.argsort(-confidence)
    bbox = bbox[sorted_ind, :]
    img_ids = img_ids[sorted_ind]

    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        gt_label = labels[img_ids[d]]
        bb = bbox[d]
        ovmax = -np.inf
        BBGT = gt_label[:, 1:5]

        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if gt_label[jmax, 6] == 0:
                tp[d] = 1.
                gt_label[jmax, 6] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


if __name__ == '__main__':
    labels = eval_truth('D:/VOC/test_imgs')
    labels = np.array(labels)
    np.save('eval_labels', labels)

