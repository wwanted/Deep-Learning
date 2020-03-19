import numpy as np
import torch
import torchvision
import glob
from torch.utils.data import Dataset
from skimage import io, transform


def read_image(path):
    img = io.imread(path)
    img = np.transpose(img, (2, 0, 1))
    return img


def normalize(img):
    img = resize(img)
    img_norm = torchvision.transforms.Normalize(mean=[0.4989, 0.4445, 0.4235], std=[0.2509, 0.2458, 0.2402])
    img = img_norm(torch.from_numpy(img).float())
    return img.numpy()


def resize(img, min_size=600, max_size=1000):
    c, h, w = img.shape
    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)
    img = transform.resize(img, (c, scale*h, scale*w), anti_aliasing=False)
    return img


def resize_bbox(bbox, insize, outsize):
    bbox = bbox.copy()
    x_scale = outsize[1] / insize[1]
    y_scale = outsize[0] / insize[0]
    bbox[:, 0] = bbox[:, 0] * x_scale
    bbox[:, 1] = bbox[:, 1] * y_scale
    bbox[:, 2] = bbox[:, 2] * x_scale
    bbox[:, 3] = bbox[:, 3] * y_scale
    return bbox


class Transform(object):
    def __init__(self):
        pass

    def __call__(self, img, bbox):
        _, h, w = img.shape
        img = normalize(img)
        _, r_h, r_w = img.shape
        scale = r_h / h
        bbox = resize_bbox(bbox, (h, w), (r_h, r_w))
        return img, bbox, scale


class GetDataset(Dataset):
    def __init__(self, path):
        self.imgpath = glob.glob(path+'/*.jpg')
        self.bboxes = np.load('bboxes.npy')
        self.labels = np.load('labels.npy')
        self.trans = Transform()

    def __getitem__(self, i):
        bbox = self.bboxes[i]
        img = read_image(self.imgpath[i])
        label = self.labels[i]
        img, bbox, scale = self.trans(img, bbox)
        return img, bbox, label, scale

    def __len__(self):
        return len(self.imgpath)











