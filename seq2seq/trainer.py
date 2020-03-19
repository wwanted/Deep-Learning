from utils import *
import numpy as np
import torch
from torchtext import data
import time
import glob
import copy


def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data.long()
        tgt = data.long()
        yield Batch(src, tgt, 0)


def subsequent_mask(size):
    sub_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(sub_mask) == 0


def tile(x, count):
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).repeat(1, count).view(*out_size).contiguous()

    return x


def copy_model(model, path='models', num=10):
    model_paths = glob.glob(path + '/model*.pth')
    models = [copy.deepcopy(model) for _ in range(num)]
    for m, path in zip(models, model_paths):
        m.load(path)

    return models


def model_average(models, src, src_mask, max_len, start_symbol, raw_prob=True):
    memories = [m.encode(src, src_mask) for m in models]
    batch_size = src.shape[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
    probs = []

    for i in range(max_len):
        for model, memory in zip(models, memories):
            out, _ = model.decode(memory, ys, src_mask, subsequent_mask(ys.shape[1]).type_as(src.data))
            prob = model.generator(out[:, -1])
            if raw_prob:
                prob = torch.exp(prob)
            probs.append(prob)

        probs = torch.stack(probs)
        mean_prob = probs.mean(dim=0)

        _, next_word = torch.max(mean_prob, dim=1)
        ys = torch.cat([ys, next_word[:, None]], dim=1)
        probs = []

    return ys


def run_epoch(train_iter, test_iter, model, opt, criterion,  pad_idx, path='', earlystopping=None, valid_step=2000):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch_ in enumerate(train_iter):
        batch = Batch(batch_.src, batch_.trg, pad=pad_idx)
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = model.loss_compute(out, batch.trg_y, criterion)

        loss.backward()
        opt.step()
        opt.optimizer.zero_grad()

        total_loss += loss.data
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 5 == 0:
            elapsed = time.time() - start
            print("Step: %d    Loss: %.4f    Tokens per Sec: %.2f    Time_used: %.2f" % (i, loss / batch.ntokens, tokens / elapsed, elapsed))
            start = time.time()
            tokens = 0

        current_step = opt.current_step

        if current_step % 100 == 0:
            with open('train_log.txt', 'a') as f:
                f.write('Total step : %d      Batch loss: %.4f      Epoch average loss: %.4f\n' % (current_step, loss / batch.ntokens, total_loss / total_tokens))

        if current_step % valid_step == 0:
            model.save(opt=opt, path=path)
            print('-----开始验证-----')
            accuracy, ppl = validate(test_iter, model, pad_idx)
            if earlystopping is not None:
                flag, best_step = earlystopping(accuracy, ppl, current_step)
                with open('valid_log.txt', 'a') as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    f.write('      Total step : %d      Accuracy: %.4f      PPL: %.4f      Best step: %d\n' % (current_step, accuracy, ppl, best_step))
                if flag:
                    break

    return total_loss / total_tokens


def run_epoch_test_with_beams(data_iter, model, pad_idx, vocab, beam_size=4, n_best=4, model_avg=False, raw_prob=True):
    start = time.time()
    total_tokens = 0
    tokens = 0
    candidate_corpus = []
    reference_corpus = []

    with torch.no_grad():
        for i, batch_ in enumerate(data_iter):
            batch = Batch(batch_.src, batch_.trg, pad=pad_idx)
            max_len = batch.src.shape[1] + 20
            batch_size = batch.src.shape[0]
            beam_search = BeamSearch(beam_size, batch_size, vocab, n_best, min_length=1, max_length=max_len, block_ngram_repeat=0)

            if model_avg:
                memory = [m.encode(batch.src, batch.src_mask) for m in model]
                memory = [tile(x, beam_size) for x in memory]
            else:
                memory = model.encode(batch.src, batch.src_mask)
                memory = tile(memory, beam_size)

            batch.src_mask = tile(batch.src_mask, beam_size)

            probs = []
            attns = []
            for step in range(max_len):
                pred = beam_search.current_predictions
                if model_avg:
                    for m, mem in zip(model, memory):
                        out, attn_temp = m.decode(mem, pred, batch.src_mask, subsequent_mask(pred.shape[1]).type_as(batch.src.data))
                        prob = m.generator(out[:, -1])
                        if raw_prob:
                            prob = torch.exp(prob)
                        probs.append(prob)
                        attns.append(attn_temp)

                    probs, attns = torch.stack(probs), torch.stack(attns)
                    mean_prob, attn = probs.mean(dim=0), attns.mean(dim=0)
                    if raw_prob:
                        log_probs = mean_prob.log()
                    else:
                        log_probs = mean_prob
                    probs = []
                    attns = []
                else:
                    out, attn = model.decode(memory, pred, batch.src_mask, subsequent_mask(pred.shape[1]).type_as(batch.src.data))
                    log_probs = model.generator(out[:, -1])

                beam_search.step(log_probs, attn)
                any_finished = beam_search.is_finished.any()
                if any_finished:
                    beam_search.update_finished()
                    if beam_search.done:
                        break
                    select_indices = beam_search.select_indices

                    if model_avg:
                        memory = [x.index_select(0, select_indices) for x in memory]
                    else:
                        memory = memory.index_select(0, select_indices)

                    batch.src_mask = batch.src_mask.index_select(0, select_indices)

            # scores = beam_search.scores
            predictions = beam_search.predictions
            best_preds = [p[0] for p in predictions]

            assert len(best_preds) == len(batch.trg_y), '预测样本与参考样本的数量不符'

            for j in range(batch_size):
                words_can = []
                words_ref = []

                candidate = best_preds[j]
                for k in range(1, len(candidate)):
                    if candidate[k] == vocab[EOS_WORD]:
                        break
                    words_can.append(vocab.itos[candidate[k]])

                reference = batch.trg_y[j]
                for v in range(len(reference)):
                    if reference[v] == vocab[EOS_WORD]:
                        break
                    words_ref.append(vocab.itos[reference[v]])

                candidate_corpus.append(words_can)
                reference_corpus.append([words_ref])

            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 5 == 0:
                elapsed = time.time() - start
                print("Step: %d     Tokens per Sec: %.2f    Time_used: %.2f" % (i, tokens / elapsed, elapsed))
                start = time.time()
                tokens = 0

    bleu = data.bleu_score(candidate_corpus, reference_corpus)

    return bleu


def run_epoch_test(data_iter, model, pad_idx, vocab, model_avg=False, raw_prob=True):
    start = time.time()
    total_tokens = 0
    tokens = 0
    candidate_corpus = []
    reference_corpus = []

    with torch.no_grad():
        for i, batch_ in enumerate(data_iter):
            batch = Batch(batch_.src, batch_.trg, pad=pad_idx)
            max_len = batch.src.shape[1] + 20
            if model_avg:
                out = model_average(model, batch.src, batch.src_mask, max_len, vocab[BOS_WORD], raw_prob=raw_prob)
            else:
                out = model.predict(batch.src, batch.src_mask, max_len, vocab[BOS_WORD])

            for j in range(out.shape[0]):
                words_can = []
                words_ref = []

                candidate = out[j]
                for k in range(1, len(candidate)):
                    if candidate[k] == vocab[EOS_WORD]:
                        break
                    words_can.append(vocab.itos[candidate[k]])

                reference = batch.trg_y[j]
                for v in range(len(reference)):
                    if reference[v] == vocab[EOS_WORD]:
                        break
                    words_ref.append(vocab.itos[reference[v]])

                candidate_corpus.append(words_can)
                reference_corpus.append([words_ref])

            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 5 == 0:
                elapsed = time.time() - start
                print("Step: %d     Tokens per Sec: %.2f    Time_used: %.2f" % (i, tokens / elapsed, elapsed))
                start = time.time()
                tokens = 0

    bleu = data.bleu_score(candidate_corpus, reference_corpus)

    return bleu


def validate(data_iter, model, pad_idx):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct_tokens = 0
    tokens = 0
    crit = torch.nn.NLLLoss(ignore_index=pad_idx, reduction='sum')

    model.eval()
    with torch.no_grad():
        for i, batch_ in enumerate(data_iter):
            batch = Batch(batch_.src, batch_.trg, pad=pad_idx)
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            scores = model.generator(out.view(-1, out.shape[-1]))
            target = batch.trg_y.contiguous().view(-1)
            loss = crit(scores, target)

            _, pred = torch.max(scores, dim=1)
            mask = (target != pad_idx)
            num_correct = pred.eq(target).masked_select(mask).sum().item()

            total_loss += loss.data
            total_tokens += batch.ntokens.item()
            total_correct_tokens += num_correct
            tokens += batch.ntokens

            if i % 5 == 0:
                elapsed = time.time() - start
                print("Valid Step: %d    Loss: %.4f    Tokens per Sec: %.2f    Time_used: %.2f" % (i, total_loss / total_tokens, tokens / elapsed, elapsed))
                start = time.time()
                tokens = 0

        accuracy = total_correct_tokens / total_tokens
        ppl = torch.exp(min(total_loss / total_tokens, 10)).item()
        model.train()

        return accuracy, ppl
