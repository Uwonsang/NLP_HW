import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path
import os
import pandas as pd

import utils, dataloader, lstm

parser = argparse.ArgumentParser(description='NMT - Seq2Seq with Attention')
""" recommend to use default settings """
# environmental settings
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--seed-num', type=int, default=0)
parser.add_argument('--save', action='store_true', default=0)
parser.add_argument('--res-dir', default='../result', type=str)
parser.add_argument('--res-tag', default='seq2seq', type=str)
# architecture
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--max-norm', type=float, default=5.0)
# hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
# option
parser.add_argument('--autoregressive', action='store_true', default=False)
parser.add_argument('--teacher-forcing', action='store_true', default=False)
parser.add_argument('--attn', action='store_true', default=True)
# etc
parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')

args = parser.parse_args()

test_batch = 1000


if not args.autoregressive:
    print(" *** Non-Autoregressive ***")

if args.teacher_forcing:
    print(" *** teacher_forcing_true ***")

if args.attn:
    print(" *** attention_true ***")

utils.set_random_seed(seed_num=args.seed_num)

use_cuda = utils.check_gpu_id(args.gpu_id)
device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')

t_start = time.time()

vocab_src = utils.read_pkl('../data/de-en/nmt_simple.src.vocab.pkl')
vocab_tgt = utils.read_pkl('../data/de-en/nmt_simple.tgt.vocab.pkl')

# recommend to split trainset
tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
                                         src_filepath='../data/de-en/nmt_simple.src.train.txt',
                                         tgt_filepath='../data/de-en/nmt_simple.tgt.train.txt',
                                         vocab=(vocab_src, vocab_tgt))
val_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
                                          src_filepath='../data/de-en/nmt_simple.src.test.txt',
                                          vocab=(tr_dataset.vocab_src, tr_dataset.vocab_tgt))

vocab_src = tr_dataset.vocab_src  # 35819
vocab_tgt = tr_dataset.vocab_tgt  # 24999
i2w_src = {v: k for k, v in vocab_src.items()}
i2w_tgt = {v: k for k, v in vocab_tgt.items()}

tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=test_batch, shuffle=False, drop_last=True, num_workers=2)

encoder = lstm.Encoder(len(vocab_src), args.hidden_size, device, args.num_layers)
if not args.attn:
    decoder = lstm.Decoder(len(vocab_tgt), args.hidden_size, device)
else:
    decoder = lstm.AttnDecoder(len(vocab_tgt), args.hidden_size, device, args.max_len, dropout_p=0.1)

utils.init_weights(encoder, init_type='uniform')
utils.init_weights(decoder, init_type='uniform')
encoder = encoder.to(device)
decoder = decoder.to(device)

""" TO DO: (masking) convert this line for masking [PAD] token """
criterion = nn.NLLLoss(ignore_index=0)

optimizer_enc = optim.Adam(encoder.parameters(), lr=args.lr)
optimizer_dec = optim.Adam(decoder.parameters(), lr=args.lr)


def train(dataloader, epoch):
    encoder.train()
    decoder.train()
    tr_loss = 0.
    correct = 0

    cnt = 0
    total_score = 0.
    prev_time = time.time()

    for idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_timestep = tgt.size(1)

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        encoder_output, encoder_state, encoder_output_length = encoder(src)
        decoder_state = encoder_state

        '([EOS], 2)'
        decoder_input = torch.tensor([args.batch_size * [2]], device=device).reshape(-1, 1)

        decoder_outputs = []
        if args.autoregressive:
            if args.teacher_forcing:
                ''' autoregressive = True + teacher_forcing = True'''
                for i in range(tgt_timestep):
                    decoder_output, decoder_state, attn_weights = decoder(decoder_input, decoder_state, encoder_output)
                    decoder_input = tgt[:, i]
                    decoder_outputs.append(decoder_output)
            else:
                ''' autoregressive = True + teacher_forcing = False'''
                for i in range(tgt_timestep):
                    decoder_output, decoder_state, attn_weights = decoder(decoder_input, decoder_state, encoder_output)
                    decoder_input = torch.argmax(decoder_output, dim=1)
                    decoder_outputs.append(decoder_output)
        else:
            for i in range(tgt_timestep):
                decoder_output, decoder_state, attn_weights = decoder(decoder_input, decoder_state, encoder_output)
                decoder_outputs.append(decoder_output)

        outputs = torch.stack(decoder_outputs, dim=1).squeeze()
        outputs = outputs.reshape(args.batch_size * args.max_len, -1)
        tgt = tgt.reshape(-1)

        loss = criterion(outputs, tgt)
        tr_loss += loss.item()
        loss.backward()

        """ TO DO: (clipping) convert this line for clipping the 'gradient < args.max_norm' """
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=args.max_norm)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.max_norm)

        optimizer_enc.step()
        optimizer_dec.step()

        # accuracy
        pred = outputs.argmax(dim=1, keepdim=True)
        pred_acc = pred[tgt != 0]
        tgt_acc = tgt[tgt != 0]
        correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

        cnt += tgt_acc.shape[0]

        # BLEU score
        score = 0.
        with torch.no_grad():
            pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
            tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
            for p, t in zip(pred, tgt):
                eos_idx = t.index(vocab_tgt['[PAD]']) if vocab_tgt['[PAD]'] in t else len(t)
                p_seq = [i2w_tgt[i[0]] for i in p][:eos_idx]
                t_seq = [i2w_tgt[i] for i in t][:eos_idx]
                k = args.k if len(t_seq) > args.k else len(t_seq)
                s = utils.bleu_score(p_seq, t_seq, k=k)
                score += s
                total_score += s

        score /= args.batch_size

        # verbose
        batches_done = (epoch - 1) * len(dataloader) + idx
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} (eta: {})".format(
            epoch, args.n_epochs, idx + 1, len(dataloader), loss, time_left), end=' ')

    tr_loss /= cnt
    tr_acc = correct / cnt
    tr_score = total_score / len(dataloader.dataset)

    return tr_loss, tr_acc, tr_score


def validate(dataloader, save=False):
	encoder.eval()
    decoder.eval()
    val_loss = 0.
    correct = 0

    cnt = 0
    total_score = 0.
    prev_time = time.time()
    pred_list = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            src_timestep = src.size(1)

			encoder_output, encoder_state, encoder_output_length = encoder(src)

            # print(encoder_state[3][0].shape)
            decoder_state = encoder_state
            decoder_input = torch.tensor([test_batch * [2]], device=device).reshape(-1, 1)


            decoder_outputs = []
            if args.autoregressive:
                if args.teacher_forcing:
                    ''' autoregressive = True + teacher_forcing = True'''
                    for i in range(src_timestep):
                        decoder_output, decoder_state, attn_weights = decoder(decoder_input, decoder_state,
                                                                              encoder_output)
                        decoder_input = tgt[:, i]
                        decoder_outputs.append(decoder_output)
                else:
                    ''' autoregressive = True + teacher_forcing = False'''
                    for i in range(src_timestep):
                        decoder_output, decoder_state, attn_weights = decoder(decoder_input, decoder_state,
                                                                              encoder_output)
                        decoder_input = torch.argmax(decoder_output, dim=1)
                        decoder_outputs.append(decoder_output)
            else:
                for i in range(src_timestep):
                    decoder_output, decoder_state, attn_weights = decoder(decoder_input, decoder_state, encoder_output)
                    decoder_outputs.append(decoder_output)

            outputs = torch.stack(decoder_outputs, dim=1).squeeze()
            print(outputs.shape)
            outputs = outputs.reshape(test_batch * args.max_len, -1)
            print(outputs.shape)
            pred = torch.reshape(outputs.argmax(dim=1, keepdim=True).cpu(), (-1,)).numpy()
            pred_list.append(pred)
            print(idx)

        pred_list = np.array(pred_list).flatten()
        # print(pred_list)
        print(len(pred_list))

        id_set = [str(i) for i in range(1, len(pred_list) + 1)]
        print(len(id_set))

        final_id = []
        for i in id_set:
            if len(i) == 1:
                final_id.append('S0000' + i)
            elif len(i) == 2:
                final_id.append('S000' + i)
            elif len(i) == 3:
                final_id.append('S00' + i)
            elif len(i) == 4:
                final_id.append('S0' + i)
            else:
                final_id.append('S' + i)

        pred_df = pd.DataFrame({
            'id': final_id,
            'pred': pred
        })

        with open(os.path.join('predict.csv'), 'w') as f:
            f.write(pred_df.to_csv(index=False))



''''main'''
for epoch in range(1, args.n_epochs + 1):
	tr_loss, tr_acc, tr_score = train(tr_dataloader, epoch)
	# {format: (loss, acc, BLEU)}
	print("tr: ({:.4f}, {:5.2f}, {:5.2f}) | ".format(tr_loss, tr_acc * 100, tr_score * 100))

validate(val_dataloader)
print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))

