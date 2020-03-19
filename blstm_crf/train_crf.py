import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import time
from 自然语言处理NLP.命名实体识别NER.BiLSTM_CRF.data_process_crf import *
from 自然语言处理NLP.命名实体识别NER.BiLSTM_CRF.utils_crf import *
from 自然语言处理NLP.命名实体识别NER.BiLSTM_CRF.blstm_crf import BiLSTM_CRF


def train(epochs=200, lr=0.05, batch_size=10, new_epoch=0, wd=0.0001, hid_dim=400, dropout=0.3):
    path = 'data/'
    model_path = 'model.pth'

    train_data = GetDataset(path)
    test_data = GetDataset(path, train=False)
    train_data_length = len(train_data)
    tag_to_id = load(path + 'tag2id.pkl')
    embeddings = np.load('data/embeddings_randnseed1.npy').astype(np.float32)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = BiLSTM_CRF(torch.from_numpy(embeddings), tag_to_id, hid_dim, dropout=dropout, window=3)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path)['net'])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    best_f1_test = 0

    for i in range(new_epoch, epochs):
        print("--------------\nEpoch %d begins!" % i)
        tic = time.time()
        tic_epoch = tic
        model.train()
        train_tags = []
        test_tags = []
        train_labels = []
        test_labels = []
        lengths = []
        sentences = []
        labels = []
        lengths_test = []
        sentences_test = []

        for step, (sentence, label) in enumerate(train_dataloader):
            lengths.append(sentence.shape[1])
            sentences.append(sentence)
            labels.append(label.squeeze())

            if (step+1) % batch_size == 0 or (step + 1) == train_data_length:
                max_len = max(lengths)
                sents = []
                for k in range(len(sentences)):
                    sent = torch.cat((sentences[k], torch.zeros((1, max_len - lengths[k]), dtype=torch.long)), dim=1)
                    sents.append(sent)
                sents = torch.cat(sents, dim=0)

                model.zero_grad()
                loss = model.neg_log_likelihood_batch(sents, labels, lengths)
                loss.backward()
                optimizer.step()

                _, train_tag = model(sentence)
                train_tags.append(train_tag)
                train_labels.append(list(label.squeeze().data.numpy()))
                lengths = []
                sentences = []
                labels = []
                
            if (step+1) % 200 == 0:
                print('training step: %d      using %.4f seconds' % (step+1, time.time() - tic))
                tic = time.time()

        train_precision, train_recall, train_f1 = cal_prf(train_tags, train_labels, tag_to_id)
        with open('train_log.txt', 'a') as f:
            f.write('Epoch: %d      Precision: %.4f     Recall: %.4f      F1: %.4f\n' % (i, train_precision, train_recall, train_f1))
        print("Epoch %d finishs!     using %.4f seconds" % (i, time.time()-tic_epoch))
            
        model.eval()
        tic_test = time.time()
        print("\nBegin to predict the results on TestData")
        with torch.no_grad():
            for k, (sentence_, label_) in enumerate(test_dataloader):
                lengths_test.append(sentence_.shape[1])
                sentences_test.append(sentence_)
                test_labels.append(list(label_.squeeze().data.numpy()))

                if (k + 1) % 200 == 0 or (k + 1) == len(test_data):
                    max_len_ = max(lengths_test)
                    sents_test = []
                    for kk in range(len(sentences_test)):
                        sent_ = torch.cat((sentences_test[kk], torch.zeros((1, max_len_ - lengths_test[kk]), dtype=torch.long)), dim=1)
                        sents_test.append(sent_)
                    sents_test = torch.cat(sents_test, dim=0)

                    test_tag = model.predict(sents_test, lengths_test)
                    test_tags.extend(test_tag)
                    print('testing step: %d      using %.4f seconds' % (k + 1, time.time() - tic))
                    tic = time.time()
                    lengths_test = []
                    sentences_test = []

        test_precision, test_recall, test_f1 = cal_prf(test_tags, test_labels, tag_to_id)
        with open('test_log.txt', 'a') as f:
            f.write('Epoch: %d      Precision: %.4f     Recall: %.4f      F1: %.4f\n' % (i, test_precision, test_recall, test_f1))

        print('testing data using %.4f seconds' % (time.time() - tic_test))

        print('----Old best acc score on test is %.4f\n' % best_f1_test)
        if test_f1 > best_f1_test:
            print("----New acc score on test is %.4f\n" % test_f1)
            best_f1_test = test_f1
            model_params = {'epoch': i+1, 'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(model_params, model_path)

        if i + 1 == int(0.8 * epochs):
            optimizer.param_groups[0]['lr'] = 0.1 * lr


if __name__ == '__main__':
    train(epochs=100, lr=0.01)

