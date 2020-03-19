import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import numpy as np
from skimage import io
import copy


def get_model():
    m = resnet50()
    m.conv1 = nn.Conv2d(3, 64, 5, 1, 2, bias=False)
    m.maxpool = nn.MaxPool2d(1, 1)
    return m


def trans_v2(img):
    img = np.asarray(img, dtype=int)

    left = np.hstack((img[:, 0][:, None], img[:, :55]))
    right = np.hstack((img[:, 1:], img[:, 55][:, None]))
    up = np.vstack((img[0][None], img[:55]))
    down = np.vstack((img[1:], img[55][None]))

    lu = np.vstack((img[0, 1:][None], img[:55, :55]))
    lu = np.hstack((img[:, 0][:, None], lu))
    ld = np.vstack((img[1:, :55], img[55, 1:][None]))
    ld = np.hstack((img[:, 0][:, None], ld))

    ru = np.vstack((img[0, :55][None], img[:55, 1:]))
    ru = np.hstack((ru, img[:, 55][:, None]))
    rd = np.vstack((img[1:, 1:], img[55, :55][None]))
    rd = np.hstack((rd, img[:, 55][:, None]))

    diff = np.stack((left, right, up, down, lu, ld, ru, rd), axis=0)
    diff = np.abs(diff - img)
    residual = np.max(diff, axis=0)

    return residual


def transfer_img(img):
    n = img.shape[0]
    img_copy = copy.deepcopy(img)
    for i in range(n):
        layer = img[i]
        for h in range(img.shape[1]):
            for w in range(img.shape[2]):
                clip = layer[max(h - 1, 0):(h + 2), max(w - 1, 0):(w + 2)]
                residual = abs(clip - layer[h, w])
                img_copy[i, h, w] = np.max(residual)
    return img_copy


class GetDataset(Dataset):
    def __init__(self, path, train=True):
        if train:
            self.imgpath = glob.glob(path+'/*.jpg')[:350000]
            self.labels = np.load('clip_labels.npy')[:350000]
        else:
            self.imgpath = glob.glob(path + '/*.jpg')[350000:]
            self.labels = np.load('clip_labels.npy')[350000:]

    def __getitem__(self, i):
        img = io.imread(self.imgpath[i])
        img = np.transpose(img, (2, 0, 1)) / 255

        label = self.labels[i]

        return torch.from_numpy(img).float(), label

    def __len__(self):
        return len(self.imgpath)


class Trainer(object):
    def __init__(self, lr=0.001, weight_decay=0):
        self.LR = lr
        self.weight_decay = weight_decay
        self.model = get_model().cuda()
        self.bn = nn.BatchNorm2d(3).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.weight_decay)

    def __call__(self, img, label):
        img = img.cuda()
        img = self.bn(img)
        self.optimizer.zero_grad()
        score = self.model(img)
        loss = F.cross_entropy(score, label.long().cuda())
        loss.backward()
        self.optimizer.step()
        loss_copy = copy.deepcopy(loss.item())

        with open('clip_losses_v2.txt', 'a') as file:
            file.write(str(loss_copy)+'\n')

        predict = torch.max(score, 1)[1].cpu().data.numpy()
        correct = (predict == label.data.numpy()).astype(int).sum()
        accuracy = correct / len(label)

        return accuracy, loss.item()

    def save(self):
        torch.save(self.model.state_dict(), 'task1_v2.pkl')

    def load(self):
        self.model.load_state_dict(torch.load('task1_v2.pkl'))

    def predict(self, img, label):
        img = img.cuda()
        img = self.bn(img)
        score = self.model(img)
        predict = torch.max(score, 1)[1].cpu().data.numpy()
        correct = (predict == label.data.numpy()).astype(int).sum()
        accuracy = correct / len(label)

        return correct, accuracy

    def task1(self, img):
        img = img.cuda()
        img = self.bn(img)
        score = self.model(img)
        predict = torch.max(score, 1)[1].cpu().data.numpy()
        characters = np.load('most_1000_characters.npy')
        pre_cha = characters[predict]

        return pre_cha
