import numpy as np
import pickle


def read_sentences_and_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [s.strip().split() for s in f if len(s.strip()) > 0]

    sentences = []
    labels = []
    sentence = []
    label = []

    for line in lines:
        sentence.append(line[0])
        label.append(line[1])

        if line[0] in ['。', '！', '？']:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []

    return sentences, labels


def get_word2id(path_list):
    words = []
    for path in path_list:
        with open(path, 'r', encoding='utf-8') as f:
            words += [s.strip().split()[0] for s in f if len(s.strip()) > 0]

    word_idx = {'_padding': 0, '_unknown': 1}
    for word in words:
        if word not in word_idx:
            word_idx[word] = len(word_idx)

    return word_idx


def sent2id(text, word2id):
    text_ids = []
    for sent in text:
        sent_id = []
        for word in sent:
            if word == '\n':
                continue
            if word in word2id:
                sent_id.append(word2id[word])
            else:
                sent_id.append(word2id['_unknown'])
        if len(sent_id) > 0:
            text_ids.append(sent_id)
        else:
            text_ids.append(word2id['_unknown'])
    assert len(text_ids) == len(text)
    return text_ids


def rand_embdding(size, vec_length, seed=None):
    if seed:
        np.random.seed(seed)
    embdding = np.random.randn(size, vec_length).astype(np.float32)
    embdding[0] *= 0.0
    return embdding


def save(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':

    path1 = 'data/train.txt'
    path2 = 'data/test.txt'
    path3 = 'data/train_dataset.pkl'
    path4 = 'data/test_dataset.pkl'
    path5 = 'data/word2id.pkl'
    path6 = 'data/tag2id.pkl'
    seed = 0

    tag2id = {'B-LOC': 0, 'I-LOC': 1, 'B-ORG': 2, 'I-ORG': 3, 'B-PER': 4, 'I-PER': 5, 'O': 6, '<START>': 7, '<STOP>': 8}
    train_text, train_labels = read_sentences_and_labels(path1)
    test_text, test_labels = read_sentences_and_labels(path2)
    word2id = get_word2id([path1, path2])
    train_sent_ids = sent2id(train_text, word2id)
    test_sent_ids = sent2id(test_text, word2id)
    train_label_ids = sent2id(train_labels, tag2id)
    test_label_ids = sent2id(test_labels, tag2id)
    train_dataset = {'sent_ids': train_sent_ids, 'labels': train_label_ids}
    test_dataset = {'sent_ids': test_sent_ids, 'labels': test_label_ids}

    print(train_dataset['sent_ids'][0])
    print(train_dataset['labels'][0])
    print(test_dataset['sent_ids'][0])
    print(test_dataset['labels'][0])
    print(word2id)
    print(tag2id)

    save(train_dataset, path3)
    save(test_dataset, path4)
    save(word2id, path5)
    save(tag2id, path6)




