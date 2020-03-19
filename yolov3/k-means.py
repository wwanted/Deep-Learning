import darknet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
import tensorflow as tf
from torchvision.models import resnet152
import torch.nn.functional as F
import torch
import glob
from skimage import io

'''
def letterbox_image(img, bbox, inp_dim):
    # resize image with unchanged aspect ratio using padding
    img_w, img_h = img.shape[1], img.shape[0]
    h, w = inp_dim
    scale = min(w / img_w, h / img_h)

    b_w, b_h = bbox[:, 2:3]*img_w, bbox[:, 3:4]*img_h
    wh_ = np.hstack((b_w, b_h))
    wh_ *= scale

    return wh_


labels1 = np.load('labels.npy')
labels2 = np.load('test_labels.npy')
imgpath1 = glob.glob('D:/VOC/imgs/*')
imgpath2 = glob.glob('D:/VOC/test_imgs/*')
box = []

for i in range(len(labels1)):
    img = io.imread(imgpath1[i])
    bbox = labels1[i].copy()
    wh = letterbox_image(img, bbox, (320, 320))
    box.append(wh)
    print(i)

for j in range(len(labels2)):
    img = io.imread(imgpath2[j])
    bbox = labels2[j].copy()
    wh = letterbox_image(img, bbox, (320, 320))
    box.append(wh)
    print(j)

box = np.concatenate(box)
print(len(box))
print(box)
np.save('box_wh', box)
'''


a = np.load('box_wh.npy')

kmean = KMeans(9, n_init=100, max_iter=2000)
x = kmean.fit_predict(a)
center = kmean.cluster_centers_

print(center)

plt.scatter(a[:, 0], a[:, 1])
plt.scatter(center[:, 0], center[:, 1])
plt.show()

