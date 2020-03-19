import torch
from utils import *
from transformer import *
from trainer import *

BATCH_SIZE = 3000

if __name__ == '__main__':
    dataset = DataloaderWMT(path='/nas/wmt14.data')
    print('-----数据预处理完成-----')
    pad_idx = dataset.pad_idx
    src_vocab = dataset.src_vocab
    tgt_vocab = dataset.tgt_vocab

    train_iter = MyIterator(dataset.train, batch_size=BATCH_SIZE, train=True)
    test_iter = MyIterator(dataset.test, batch_size=BATCH_SIZE, train=False)
    print('-----Batch准备完成-----')

    model = Transformer(len(src_vocab), len(tgt_vocab), layers=6, dropout=0.3)
    noamopt = NoamOpt(model.d_model, factor=1, warm_up=4000,
                      optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = LabelSmoothing(tgt_vocab_size=len(tgt_vocab), padding_idx=pad_idx, smoothing=0.1)
    earlystopping = EarlyStopping(10)

    # model.load('/nas/model_0.pth')
    # noamopt.load('/nas/optimizer_0.pth')
    print('-----模型已创建完成-----')

    for epoch in range(0, 20):
        print('-----Epoch %d 开始训练-----' % epoch)
        with open('train_log.txt', 'a') as f:
            f.write('-----Epoch %d 开始训练-----\n\n' % epoch)
        model.train()
        run_epoch(train_iter, test_iter, model, noamopt, criterion, pad_idx,
                  earlystopping=earlystopping, valid_step=2000)
        if earlystopping.stopped:
            break

        model.eval()
        print('-----Epoch %d 开始测试-----' % epoch)
        with open('train_log.txt', 'a') as f:
            f.write('\n-----Epoch %d 开始测试-----\n\n' % epoch)
        bleu = run_epoch_test(test_iter, model, pad_idx, tgt_vocab)
        print('Epoch: %d    BLEU: %.4f' % (epoch, bleu))
        with open('train_log.txt', 'a') as f:
            f.write('Epoch: %d    BLEU: %.4f\n\n' % (epoch, bleu))



