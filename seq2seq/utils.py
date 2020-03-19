import numpy as np
import torch
from torchtext import data, datasets
import spacy

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def subsequent_mask(size):
    sub_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(sub_mask) == 0


def batch_size_fn_(new, count, sofar):
    global max_in_batch
    if count == 1:
        max_in_batch = 0
    max_in_batch = max(max_in_batch,  len(new.src), len(new.trg) + 2)
    return count * max_in_batch


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class MyIterator(data.Iterator):
    def __init__(self, dataset, batch_size, sort_key=lambda x: (len(x.src), len(x.trg)), device=None, batch_size_fn=batch_size_fn_,
                 train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=False):
        super(MyIterator, self).__init__(dataset, batch_size, sort_key, device, batch_size_fn, train, repeat, shuffle, sort, sort_within_batch)

    def create_batches(self):
        if self.train:
            self.batches = self.pool(self.data(), self.random_shuffler)
        else:
            self.batches = data.batch(self.data(), self.batch_size, self.batch_size_fn)

    def pool(self, d, random_shuffler):
        for p in data.batch(d, self.batch_size * 10):
            p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
            for b in random_shuffler(list(p_batch)):
                yield b


class Dataloader(object):
    def __init__(self, path=None, min_freq=2):

        self.src_func = tokenize_de if path is None else lambda x: x
        self.tgt_func = tokenize_en if path is None else lambda x: x
        self.data = torch.load(path) if path is not None else None
        self.min_freq = min_freq

        self.SRC = data.Field(tokenize=self.src_func, pad_token=BLANK_WORD, lower=True, batch_first=True)
        self.TGT = data.Field(tokenize=self.tgt_func, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD, lower=True, batch_first=True)

        self.train = None
        self.test = None

    def build_vocab(self):
        self.SRC.build_vocab(self.train.src, min_freq=self.min_freq)
        self.TGT.build_vocab(self.train.trg, min_freq=self.min_freq)

    def save(self, path):
        data = {'train_data': self.train.examples, 'test_data': self.test.examples, 'src': self.SRC, 'tgt': self.TGT}
        torch.save(data, path)

    def load_examples(self):
        self.train.examples = self.data['train_data']
        self.test.examples = self.data['test_data']

    def load_fields(self):
        self.SRC = self.data['src']
        self.TGT = self.data['tgt']
        self.SRC.tokenize = self.src_func
        self.TGT.tokenize = self.tgt_func


class DataloaderIwslt(Dataloader):
    def __init__(self, path=None, min_freq=2, max_len=100):
        super(DataloaderIwslt, self).__init__(path=path, min_freq=min_freq)
        if path:
            self.load_fields()
        self.train, val, self.test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(self.SRC, self.TGT),
                                                           root='data', filter_pred=lambda x: len(vars(x)['src']) <= max_len and len(vars(x)['trg']) <= max_len)
        self.test.examples.extend(val.examples)
        if path:
            self.load_examples()
        else:
            self.build_vocab()

        self.pad_idx = self.TGT.vocab.stoi[BLANK_WORD]
        self.src_vocab = self.SRC.vocab
        self.tgt_vocab = self.TGT.vocab


class DataloaderWMT(Dataloader):
    def __init__(self, path=None, min_freq=5):
        super(DataloaderWMT, self).__init__(path=path, min_freq=min_freq)
        if path:
            self.load_fields()
        self.train, self.test = datasets.WMT14.splits(exts=('.de', '.en'), fields=(self.SRC, self.TGT),
                                                      root='data', train='newstest2014', validation=None, test='newstest2013')
        if path:
            self.load_examples()
        else:
            self.build_vocab()

        self.pad_idx = self.TGT.vocab.stoi[BLANK_WORD]
        self.src_vocab = self.SRC.vocab
        self.tgt_vocab = self.TGT.vocab


class BeamSearch(object):
    def __init__(self, beam_size, batch_size, vocab, n_best, min_length, max_length,
                 block_ngram_repeat, exclusion_tokens=frozenset(), cov_pen=True, len_pen=True):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.n_best = n_best

        self.pad = vocab[BLANK_WORD]
        self.bos = vocab[BOS_WORD]
        self.eos = vocab[EOS_WORD]

        self.top_beam_finished = torch.zeros(batch_size, dtype=torch.bool)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self._cov_pen = cov_pen
        self._len_pen = len_pen
        self.alpha = 0.2
        self.beta = 0.4
        self._coverage = None

        self.alive_seq = torch.full((batch_size * beam_size, 1), self.bos, dtype=torch.long)
        self.is_finished = torch.zeros((batch_size, beam_size), dtype=torch.bool)

        self._beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long)
        self._batch_index = torch.empty((batch_size, beam_size), dtype=torch.long)

        self.topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size)
        self.topk_scores = torch.empty((batch_size, beam_size), dtype=torch.float)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long)

        self.select_indices = None
        self.done = False

        self.hypotheses = [[] for _ in range(batch_size)]
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]

        self.min_length = min_length
        self.max_length = max_length

        self.block_ngram_repeat = block_ngram_repeat
        self.forbidden_tokens = [dict() for _ in range(batch_size * beam_size)]
        self.exclusion_tokens = {vocab[t] for t in exclusion_tokens}

        self.alive_seq = torch.full([self.batch_size * self.beam_size, 1], self.bos, dtype=torch.long)
        self.is_finished = torch.zeros([self.batch_size, self.beam_size], dtype=torch.uint8)

    def __len__(self):
        return self.alive_seq.shape[1]

    def length_penalty(self, cur_len):
        if self._len_pen:
            return ((5 + cur_len) / 6.0) ** self.alpha
        else:
            return 1.0

    def coverage_penalty(self, cov):
        cov = torch.clamp(cov, min=0.05, max=1.0)
        penalty = -cov.log().sum(-1)
        return self.beta * penalty

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    @property
    def current_predictions(self):
        return self.alive_seq

    @property
    def batch_offset(self):
        return self._batch_offset

    def block_ngram_repeats(self, log_probs):
        if self.block_ngram_repeat <= 0:
            return

        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat - 1
        for path_idx in range(self.alive_seq.shape[0]):
            current_ngram = tuple(self.alive_seq[path_idx, -n:].tolist())
            forbidden_tokens = self.forbidden_tokens[path_idx].get(current_ngram, None)
            if forbidden_tokens is not None:
                log_probs[path_idx, list(forbidden_tokens)] = -1e20

    def maybe_update_forbidden_tokens(self):
        if self.block_ngram_repeat <= 0:
            return

        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat
        forbidden_tokens = list()
        for path_idx, seq in zip(self.select_indices, self.alive_seq):
            forbidden_tokens.append(dict(self.forbidden_tokens[path_idx]))

            current_ngram = tuple(seq[-n:].tolist())
            if set(current_ngram) & self.exclusion_tokens:
                continue

            forbidden_tokens[-1].setdefault(current_ngram[:-1], set())
            forbidden_tokens[-1][current_ngram[:-1]].add(current_ngram[-1])

        self.forbidden_tokens = forbidden_tokens

    def step(self, log_probs, attn):
        vocab_size = log_probs.size(-1)
        _B = log_probs.shape[0] // self.beam_size
        step = len(self)
        self.ensure_min_length(log_probs)

        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)
        len_penalty = self.length_penalty(step + 1)
        curr_scores = log_probs / len_penalty

        self.block_ngram_repeats(curr_scores)

        self.topk_scores, self.topk_ids = torch.topk(curr_scores.view(_B, -1), self.beam_size, dim=-1)
        self.topk_log_probs = self.topk_scores * len_penalty

        self._batch_index = torch.div(self.topk_ids, vocab_size)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)

        self.alive_seq = torch.cat([self.alive_seq.index_select(0, self.select_indices), self.topk_ids.view(_B * self.beam_size, 1)], -1)

        self.maybe_update_forbidden_tokens()

        if self._cov_pen:
            current_attn = attn.index_select(0, self.select_indices)
            if step == 1:
                self._coverage = current_attn
            else:
                self._coverage = self._coverage.index_select(0, self.select_indices)
                self._coverage += current_attn

            cov_penalty = self.coverage_penalty(self._coverage)
            self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        _B_old = self.topk_log_probs.shape[0]
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)

        self.top_beam_finished = self.top_beam_finished | self.is_finished[:, 0].bool()
        predictions = self.alive_seq.view(_B_old, self.beam_size, -1)

        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)

            for j in finished_hyp:
                self.hypotheses[b].append([self.topk_scores[i, j], predictions[i, j, 1:]])

            finish_flag = self.top_beam_finished[i] != 0

            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
            else:
                non_finished_batch.append(i)

        if len(non_finished_batch) == 0:
            self.done = True
            return

        non_finished = torch.tensor(non_finished_batch)
        _B_new = non_finished.shape[0]

        self.top_beam_finished = self.top_beam_finished.index_select(0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished).view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        if self._cov_pen:
            self._coverage = self._coverage.view(_B_old, self.beam_size, -1).\
                index_select(0, non_finished).view(_B_new * self.beam_size, -1)


class EarlyStopping(object):
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.stalled_tolerance = self.tolerance
        self.current_tolerance = self.tolerance
        self.current_step_best = 0

        self.stopped = False
        self.best_accuracy = float('-inf')
        self.best_ppl = float('inf')

    def __call__(self, acc, ppl, step):
        if acc > self.best_accuracy and ppl < self.best_ppl:
            self.current_step_best = step
            self.best_accuracy = acc
            self.best_ppl = ppl
            self.current_tolerance = self.tolerance
            self.stalled_tolerance = self.tolerance
        elif acc < self.best_accuracy and ppl > self.best_ppl:
            self.current_tolerance -= 1
        else:
            self.stalled_tolerance -= 1

        if self.current_tolerance <= 0 or self.stalled_tolerance <= 0:
            self.stopped = True

        return self.stopped, self.current_step_best








