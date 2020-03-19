import torch
import torch.nn as nn
import numpy as np

# torch.manual_seed(1)
START_TAG = '<START>'
STOP_TAG = '<STOP>'


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_matrix(vec):
    max_score = torch.max(vec, dim=1)[0]
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.view(-1, 1)), dim=1))


def local_attention_id(size, window):
    prior = []

    for k in range(window):
        prior.extend([window + k, window - k - 1])

    prior = torch.tensor(prior, dtype=torch.long)

    ind = torch.arange(size).view(-1, 1)
    ind = ind.expand(-1, 2 * window)
    ind_bias = np.arange(1, window + 1, dtype=np.int64)
    ind_bias = np.concatenate((-ind_bias[::-1], ind_bias), axis=0)
    ind_bias = torch.from_numpy(ind_bias)
    ind = ind + ind_bias

    for i in range(window):
        neg_ind = ind[i][ind[i] < 0]
        prior_ind = ind[i][prior]
        prior_ind = prior_ind[prior_ind >= 0]
        assert len(neg_ind) <= len(prior_ind)
        ind[i][:len(neg_ind)] = prior_ind[:len(neg_ind)]

        over_ind = ind[-(i+1)][ind[-(i+1)] >= size]
        prior_ind_over = ind[-(i+1)][prior]
        prior_ind_over = prior_ind_over[prior_ind_over < size]
        assert len(over_ind) <= len(prior_ind_over)
        ind[-(i+1)][-len(over_ind):] = prior_ind_over[:len(over_ind)]

    return ind


class BiLSTM_CRF(nn.Module):
    def __init__(self, embeddings, tag_to_ix, hidden_dim, dropout=0.5, window=3):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embeddings.shape[1]
        self.hidden_dim = hidden_dim
        self.vocab_size = embeddings.shape[0]
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.window = window

        self.word_embeds = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)
        self.ht2query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            emit_score = feat.view(self.tagset_size, 1)
            all_score = forward_var + self.transitions + emit_score
            forward_var = log_sum_exp_matrix(all_score).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence).view(1, -1, self.embedding_dim)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[0]
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_lstm_features_batch(self, sentences, lengths):
        # self.hidden_batch = (self.hidden[0].expand(-1, len(sentences), -1), self.hidden[1].expand(-1, len(sentences), -1))
        embeds = self.word_embeds(sentences)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = []
        for i in range(len(sentences)):
            input_x = embeds[i, :lengths[i]]
            ht = lstm_out[i, :lengths[i]]
            query = self.ht2query(ht).view(lengths[i], -1, 1)
            attention_key_ = self.key(torch.cat((input_x, ht), dim=1))
            attention_value_ = self.value(torch.cat((input_x, ht), dim=1))
            
            attention_id = local_attention_id(lengths[i], self.window)
            attention_key = attention_key_[attention_id]
            attention_value = attention_value_[attention_id]
            attention_score = torch.bmm(attention_key, query) / np.sqrt(query.shape[1])
            attention_prob = nn.functional.softmax(attention_score, dim=1)
            attention_prob = nn.functional.dropout(attention_prob, p=0.2, training=self.training)
            attention = torch.sum(attention_prob * attention_value, dim=1)

            feat = torch.cat((ht, attention), dim=1)
            feat = self.dropout(feat)
            feat = self.hidden2tag(feat)
            lstm_feats.append(feat)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            tag_var = forward_var + self.transitions
            viterbivars_t, best_tag_id = torch.max(tag_var, dim=1)
            forward_var = (viterbivars_t + feat).view(1, -1)
            bptrs_t = list(best_tag_id.data.numpy())

            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def neg_log_likelihood_batch(self, sentences, tags, lengths):
        feats = self._get_lstm_features_batch(sentences, lengths)
        loss_score = torch.zeros(1)
        for i in range(len(sentences)):
            feat = feats[i]
            tag = tags[i]

            forward_score = self._forward_alg(feat)
            gold_score = self._score_sentence(feat, tag)
            loss_score += (forward_score - gold_score)
        return loss_score / len(sentences)

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features_batch(sentence, [sentence.shape[1]])

        score, tag_seq = self._viterbi_decode(lstm_feats[0])
        return score, tag_seq
    
    def predict(self, sentences, lengths):
        lstm_feats = self._get_lstm_features_batch(sentences, lengths)
        tag_seqs = []
        for i in range(len(sentences)):
            feat = lstm_feats[i]
    
            score, tag_seq = self._viterbi_decode(feat)
            tag_seqs.append(tag_seq)
        return tag_seqs



