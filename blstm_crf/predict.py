from 自然语言处理NLP.命名实体识别NER.BiLSTM_CRF.data_process_crf import *
from 自然语言处理NLP.命名实体识别NER.BiLSTM_CRF.utils_crf import *
from 自然语言处理NLP.命名实体识别NER.BiLSTM_CRF.blstm_crf import BiLSTM_CRF
from torch.utils.data import DataLoader
import os
import time


def predict(batch_size=100, hid_dim=400, dropout=0.3):
    path = 'data/'
    model_path = 'model.pth'

    test_data = GetDataset(path, train=False)
    word2id = load(path + 'word2id.pkl')
    id2word = {}
    for word, id1 in word2id.items():
        id2word[id1] = word

    tag_to_id = load(path + 'tag2id.pkl')
    id2tag = {}
    for tag, id2 in tag_to_id.items():
        id2tag[id2] = tag

    embeddings = np.load('data/embeddings_randnseed1.npy').astype(np.float32)

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = BiLSTM_CRF(torch.from_numpy(embeddings), tag_to_id, hid_dim, dropout=dropout, window=3)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path)['net'])

    print("------------------------------\nBegin to predict the results!" )
    tic = time.time()
    start_time = tic
    model.eval()

    test_tags = []
    test_labels = []
    lengths_test = []
    sentences_test = []
    sentences_predict = []

    with torch.no_grad():
        for k, (sentence, label) in enumerate(test_dataloader):
            lengths_test.append(sentence.shape[1])
            sentences_test.append(sentence)
            test_labels.append(list(label.squeeze().data.numpy()))
            sentences_predict.append(list(sentence.squeeze().data.numpy()))

            if (k + 1) % batch_size == 0 or (k + 1) == len(test_data):
                max_len = max(lengths_test)
                sents_test = []
                for kk in range(len(sentences_test)):
                    sent = torch.cat((sentences_test[kk], torch.zeros((1, max_len - lengths_test[kk]), dtype=torch.long)), dim=1)
                    sents_test.append(sent)
                sents_test = torch.cat(sents_test, dim=0)

                test_tag = model.predict(sents_test, lengths_test)
                test_tags.extend(test_tag)
                lengths_test = []
                sentences_test = []

            if (k + 1) % 200 == 0:
                print('predicting step: %d      using %.4f seconds' % (k+1, time.time() - tic))
                tic = time.time()
    print('------------------------------')
    test_precision, test_recall, test_f1 = cal_prf(test_tags, test_labels, tag_to_id)
    with open('results.txt', 'w') as f:
        for i, sentence_id in enumerate(sentences_predict):
            predict_tag = test_tags[i]
            gold_label = test_labels[i]
            assert len(sentence_id) == len(predict_tag) == len(gold_label)
            for word, lab, tag in zip(sentence_id, gold_label, predict_tag):
                f.write(id2word[word] + ' ' + id2tag[lab] + ' ' + id2tag[tag] + '\n')
            f.write('\n')
    print('------------------------------')
    print('predict data using %.4f seconds' % (time.time() - start_time))

