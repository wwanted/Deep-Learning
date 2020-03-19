from torch.utils.data import Dataset
import torch
import pickle


def load(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def cal_prf(tags, labels, tag_to_id):
    assert len(tags) == len(labels)
    right = 0
    gold = 0
    pred = 0
    for i in range(len(tags)):
        t = tags[i]
        l = labels[i]
        assert len(t) == len(l)
        for j in range(len(t)):
            tag = t[j]
            label = l[j]
            if label != tag_to_id['O']:
                gold += 1
                if tag == label:
                    right += 1
            if tag != tag_to_id['O']:
                pred += 1

    precision = right / pred if pred > 0 else 0
    recall = right / gold if gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print('Precision: %.4f     Recall: %.4f      F1: %.4f' % (precision, recall, f1))

    return precision, recall, f1


class GetDataset(Dataset):
    def __init__(self, path, train=True):
        if train:
            self.data = load(path+'train_dataset.pkl')
            self.sentences = self.data['sent_ids']
            self.labels = self.data['labels']
        else:
            self.data = load(path + 'test_dataset.pkl')
            self.sentences = self.data['sent_ids'][:1000]
            self.labels = self.data['labels'][:1000]

        assert len(self.sentences) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        sentence = torch.tensor(self.sentences[i], dtype=torch.long)
        label = torch.tensor(self.labels[i], dtype=torch.long)
        assert len(sentence) == len(label)

        return sentence, label


def change_lr(opt, epoch, warm_up=10, factor=1):
    fac = factor
    epoch += 1

    if 50 < epoch <= 100:
        fac = factor / 2
    elif 100 < epoch <= 150:
        fac = factor / 4
    elif 150 < epoch <= 200:
        fac = factor / 8
    elif epoch > 200:
        fac = factor / 16

    rate = fac * 0.16 * min(epoch ** -0.5, epoch * warm_up ** -1.5)
    for p in opt.param_groups:
        p['lr'] = rate

