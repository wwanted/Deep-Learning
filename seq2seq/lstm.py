# from transformer import *
from 自然语言处理NLP.Transformer.transformer import *


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, pad_idx, position):
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        self.position = position
        self.d_model = d_model

    def forward(self, x):
        x = self.embeddings(x) * math.sqrt(self.d_model)
        x = self.position(x)
        return x


class GlobalAttention(nn.Module):
    def __init__(self, dim_size):
        super(GlobalAttention, self).__init__()
        self.dim_size = dim_size
        self.w_in = nn.Linear(dim_size, dim_size, bias=False)
        self.w_out = nn.Linear(dim_size*2, dim_size, bias=False)

    def forward(self, tgt, memory, src_mask):
        batch, src_l, dim = memory.size()
        batch_, tgt_l, dim_ = tgt.size()
        assert batch == batch_
        assert dim == dim_
        assert self.dim_size == dim

        tgt = self.w_in(tgt)
        attn_temp = torch.bmm(tgt, memory.transpose(1, 2))
        attn_temp = attn_temp.masked_fill(src_mask == 0, -1e20)
        attns = F.softmax(attn_temp, dim=-1)

        cell_state = torch.bmm(attns, memory)
        concat = torch.cat([cell_state, tgt], 2)
        attn_h = torch.tanh(self.w_out(concat))

        return attn_h, attns


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, embeddings, dropout=0.1, bidirectional=True, use_bridge=True,):
        super(Encoder, self).__init__()
        assert embeddings is not None
        assert hidden_size % 2 == 0

        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.embeddings = embeddings
        self.num_layers = num_layers

        self.lstm_encode = nn.LSTM(input_size=embeddings.d_model, hidden_size=self.hidden_size,
                                   num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.use_bridge = use_bridge
        self.total_dim = hidden_size * num_layers
        if self.use_bridge:
            self.bridge_c = nn.Linear(self.total_dim, self.total_dim, bias=True)
            self.bridge_h = nn.Linear(self.total_dim, self.total_dim, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embeddings(x)
        memory, (hn, cn) = self.lstm_encode(x)

        if self.use_bridge:
            hn = self.bridge_h(hn.transpose(0, 1).contiguous().view(-1, self.total_dim))
            cn = self.bridge_c(cn.transpose(0, 1).contiguous().view(-1, self.total_dim))
            hn = hn.view(batch_size, self.num_layers, -1).transpose(0, 1).contiguous()
            cn = cn.view(batch_size, self.num_layers, -1).transpose(0, 1).contiguous()
        else:
            hn = hn.view(self.num_layers, -1, batch_size, self.hidden_size).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)
            cn = cn.view(self.num_layers, -1, batch_size, self.hidden_size).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)

        return memory, (hn, cn)


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, embeddings, dropout=0.1, attentional=True):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.attentional = attentional
        if self.attentional:
            self.attn_layer = GlobalAttention(hidden_size)

        self.lstm_decode = nn.LSTM(input_size=embeddings.d_model, hidden_size=self.hidden_size,
                                   num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, memory, src_mask, encoder_state):
        attns = {}
        x = self.embeddings(x)
        x, final_state = self.lstm_decode(x, encoder_state)
        if not self.attentional:
            dec_out = x
        else:
            dec_out, p_attn = self.attn_layer(x, memory, src_mask)
            attns['std'] = p_attn

        dec_out = self.dropout(dec_out)

        return dec_out, attns, final_state


class LSTM(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, pad_idx, d_model=512, dropout=0.1, layers=2, attn=True):
        super(LSTM, self).__init__()
        self.position = PositionalEncoding(d_model, dropout)
        self.src_embeds = Embeddings(src_vocab, d_model, pad_idx, self.position)
        self.tgt_embeds = Embeddings(tgt_vocab, d_model, pad_idx, self.position)

        self.encoder = Encoder(d_model, layers, self.src_embeds, dropout=dropout)
        self.decoder = Decoder(d_model, layers, self.tgt_embeds, dropout=dropout, attentional=attn)

        self.generator = Generator(d_model, tgt_vocab)
        self.d_model = d_model
        self.check_num = 0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, *args):
        memory, encoder_state = self.encoder(src)
        dec_out, attns, final_state = self.decoder(tgt, memory, src_mask, encoder_state)

        return dec_out

    def loss_compute(self, x, y, criterion):
        x = self.generator(x)
        loss = criterion(x.contiguous().view(-1, x.shape[-1]), y.contiguous().view(-1))

        return loss

    def predict(self, src, src_mask, max_len, start_symbol):
        memory, hc_state = self.encoder(src)
        batch_size = src.shape[0]
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)

        for i in range(max_len):
            out, _, hc_state = self.decoder(ys[:, -1:], memory, src_mask, hc_state)
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

