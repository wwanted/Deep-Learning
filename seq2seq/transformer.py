import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def subsequent_mask(size):
    sub_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(sub_mask) == 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e20)
        attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        return torch.matmul(p_attn, value), attn

    def forward(self, query, key, value, mask=None):
        n_batches = query.shape[0]

        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, attn = self.attention(query, key, value,  mask=mask)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        return self.linears[-1](x), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        attention, _ = self.attn(x, x, x, mask)
        x = x + self.dropout(attention)
        x = self.norm_attn(x)
        forward_var = self.feed_forward(x)
        x = x + self.dropout(forward_var)
        x = self.norm_ffn(x)
        return x


class EncoderLayerEasy(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(EncoderLayerEasy, self).__init__()
        self.attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x1 = self.norm_attn(x)
        x2, _ = self.attn(x1, x1, x1, mask)
        x3 = x + self.dropout(x2)
        x4 = self.norm_ffn(x3)
        x5 = self.feed_forward(x4)
        x6 = x3 + self.dropout(x5)
        return x6


class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.src_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm_self_attn = LayerNorm(d_model)
        self.norm_src_attn = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        self_attention, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attention)
        x = self.norm_self_attn(x)
        src_attention, attn = self.src_attn(x, m, m, src_mask)
        x = x + self.dropout(src_attention)
        x = self.norm_src_attn(x)
        forward_var = self.feed_forward(x)
        x = x + self.dropout(forward_var)
        x = self.norm_ffn(x)

        return x, attn[:, 0, -1, :]


class DecoderLayerEasy(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(DecoderLayerEasy, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.src_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm_self_attn = LayerNorm(d_model)
        self.norm_src_attn = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x1 = self.norm_self_attn(x)
        x2, _ = self.self_attn(x1, x1, x1, tgt_mask)
        x3 = x + self.dropout(x2)
        x4 = self.norm_src_attn(x3)
        x5, attn = self.src_attn(x4, m, m, src_mask)
        x6 = x3 + self.dropout(x5)
        x7 = self.norm_ffn(x6)
        x8 = self.feed_forward(x7)
        x9 = x6 + self.dropout(x8)

        return x9, attn[:, 0, -1, :]


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class LabelSmoothing(nn.Module):
    def __init__(self, tgt_vocab_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = tgt_vocab_size
        self.true_dist = None

    def forward(self, x, target):
        assert x.shape[1] == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt(object):
    def __init__(self, model_size, factor, warm_up, optimizer):
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.factor = factor
        self.model_size = model_size
        self.current_step = 0
        self._rate = self.rate(1)

    def step(self):
        self.current_step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self.current_step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warm_up ** (-1.5)))

    def load(self, path):
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_step = checkpoint['step']


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, h=8, d_model=512, d_ff=2048, dropout=0.1, layers=6, easy_mode=True):
        super(Transformer, self).__init__()
        if easy_mode:
            self.encoder = nn.ModuleList([EncoderLayerEasy(h, d_model, d_ff, dropout) for _ in range(layers)])
            self.decoder = nn.ModuleList([DecoderLayerEasy(h, d_model, d_ff, dropout) for _ in range(layers)])
        else:
            self.encoder = nn.ModuleList([EncoderLayer(h, d_model, d_ff, dropout) for _ in range(layers)])
            self.decoder = nn.ModuleList([DecoderLayer(h, d_model, d_ff, dropout) for _ in range(layers)])

        self.norm_encoder = LayerNorm(d_model)
        self.norm_decoder = LayerNorm(d_model)
        self.src_embeds = nn.Embedding(src_vocab, d_model)
        self.tgt_embeds = nn.Embedding(tgt_vocab, d_model)
        self.position = PositionalEncoding(d_model, dropout)

        self.generator = Generator(d_model, tgt_vocab)

        self.d_model = d_model
        self.easy_mode = easy_mode
        self.check_num = 0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):

        x = self.encode(src, src_mask)
        y, _ = self.decode(x, tgt, src_mask, tgt_mask)

        return y

    def encode(self, src, src_mask):
        x = self.src_embeds(src) * math.sqrt(self.d_model)
        x = self.position(x)

        for layer_x in self.encoder:
            x = layer_x(x, src_mask)

        if self.easy_mode:
            x = self.norm_encoder(x)

        return x

    def decode(self, memory, tgt, src_mask, tgt_mask):
        y = self.tgt_embeds(tgt) * math.sqrt(self.d_model)
        y = self.position(y)
        attn = None

        for layer_y in self.decoder:
            y, attn = layer_y(y, memory, src_mask, tgt_mask)

        if self.easy_mode:
            y = self.norm_decoder(y)

        return y, attn

    def loss_compute(self, x, y, criterion):
        x = self.generator(x)
        loss = criterion(x.contiguous().view(-1, x.shape[-1]), y.contiguous().view(-1))

        return loss

    def predict(self, src, src_mask, max_len, start_symbol):
        memory = self.encode(src, src_mask)
        batch_size = src.shape[0]
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)

        for i in range(max_len):
            out, _ = self.decode(memory, ys, src_mask, subsequent_mask(ys.shape[1]).type_as(src.data))
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word[:, None]], dim=1)

        return ys

    def save(self, opt, path='', max_n=20):
        check_idx = self.check_num % max_n
        model_path = path + 'model_{:0>2d}.pth'.format(check_idx)
        opt_path = path + 'optimizer_{:0>2d}.pth'.format(check_idx)
        self.check_num += 1
        model_state = {'net': self.state_dict(), 'checkpoint': self.check_num}
        opt_state = {'optimizer': opt.optimizer.state_dict(), 'step': opt.current_step}
        torch.save(model_state, model_path)
        torch.save(opt_state, opt_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['net'])
        self.check_num = checkpoint['checkpoint']







