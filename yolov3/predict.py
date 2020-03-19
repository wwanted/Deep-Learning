import time
import torch
import train
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from skimage import io
import skimage
import glob
import random


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
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

    return canvas


def write(x, img, colors):
    c1 = tuple(x[1:3].astype(int))
    c2 = tuple(x[3:5].astype(int))
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, 1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    return img


class GetDataset(Dataset):
    def __init__(self, path, size=320):
        self.size = size
        self.imgpath = glob.glob(path+'/*')

    def __getitem__(self, i):
        img = io.imread(self.imgpath[i])

        img = letterbox_image(img, (self.size, self.size))
        img = np.transpose(img, (2, 0, 1)) / 255

        return torch.from_numpy(img).float(),i

    def __len__(self):
        return len(self.imgpath)


if __name__ == '__main__':

    time1 = time.time()
    path = 'test'
    dataset = GetDataset(path)
    imgpath = glob.glob(path + '/*')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    trainer = train.Trainer()
    time2 = time.time()
    print('已创建模型：%.4f秒' % (time2 - time1))
    time3 = time.time()
    print('已导入参数：%.4f秒' % (time3 - time2))
    print("----------------------------------------------------------")
    classes = load_classes('data/coco.names')
    size = trainer.model.input_size
    img_num = 1

    for i, (imgs, index) in enumerate(dataloader):
        time4 = time.time()
        batch_size = len(imgs)
        output = trainer.predict_no_label(imgs)
        index = index.data.numpy().astype(int)
        time5 = time.time()

        if type(output) == int:
            print("Batch{:0>3d}没有发现目标物体".format(i))
            output = np.zeros((1, 8))
            output[:, 0] = -1

        else:
            for j in range(batch_size):
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == j]
                print("Batch{:0>3d}第 {:d} 张图片预测时间：{:.2f}秒".format(i, j+1, (time5 - time4)/batch_size))
                if len(objs) == 0:
                    print("{:s}".format("没有发现目标物体:"))
                else:
                    print("{:s} {:s}".format("发现目标物体:", " ".join(objs)))
                print("----------------------------------------------------------")

            output = output.data.numpy()

        for k in range(len(index)):
            org_img = io.imread(imgpath[index[k]])
            ind = np.where(output[:, 0] == k)
            bbox = output[ind]
            if len(bbox) == 0:
                io.imsave('output/predict_{:0>6d}.jpg'.format(img_num), org_img)
                img_num += 1
                continue

            img_w, img_h = org_img.shape[1], org_img.shape[0]
            h = w = size
            scale = min(w / img_w, h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            dx = (w - new_w) // 2
            dy = (h - new_h) // 2

            bbox[:, [1, 3]] = (bbox[:, [1, 3]] - dx) / scale
            bbox[:, [2, 4]] = (bbox[:, [2, 4]] - dy) / scale
            bbox[:, [1, 3]] = bbox[:, [1, 3]].clip(0, img_w-1)
            bbox[:, [2, 4]] = bbox[:, [2, 4]].clip(0, img_h-1)

            colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]
            for box in bbox:
                org_img = write(box, org_img, colors)

            io.imsave('output/predict_{:0>6d}.jpg'.format(img_num), org_img)
            img_num += 1

        time6 = time.time()
        print("Batch{:0>3d}共 {:d} 张图片已打印输出，消耗时间：{:.2f}秒".format(i, batch_size, time6 - time5))
        print("...........................................................")
        print("...........................................................")









