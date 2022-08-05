import torch
import torch.nn as nn
from model.sub_layers import Embedding_Layer, Encoder, Decoder
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self, vocab, num_token, max_seq_len, dim_model, n_head=8, dim_hidden=2048, d_prob=0.1, n_enc_layer=6, n_dec_layer=6, device=device):
        super(Transformer, self).__init__()

        self.device = device
        self.num_token = num_token
        self.max_seq_len = max_seq_len
        self.src_pad_idx = vocab['[PAD]']
        self.trg_pad_idx = vocab['[PAD]']
        self.trg_sos_idx = vocab['[SOS]']

        self.embed = Embedding_Layer(num_token=num_token, dim_model=dim_model, max_seq_len=max_seq_len, device=device)

        self.encoder = Encoder(d_model=dim_model,
                               n_head=n_head,
                               max_len=max_seq_len,
                               ffn_hidden=dim_hidden,
                               enc_voc_size=num_token,
                               n_layers=n_enc_layer,
                               device=device)

        self.decoder = Decoder(d_model=dim_model,
                               n_head=n_head,
                               max_len=max_seq_len,
                               ffn_hidden=dim_hidden,
                               dec_voc_size=num_token,
                               n_layers=n_dec_layer,
                               device=device)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src, src)

        src_trg_mask = self.make_pad_mask(tgt, src)
        tgt_mask = self.make_pad_mask(tgt, tgt) * self.make_sequence_mask(tgt, tgt)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_trg_mask)
        output = self.softmax(output)

        return output

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)

    ''''''''''''''''''''''''''
    '''2.1 Pad mask'''
    ''''''''''''''''''''''''''
    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2) # batch_size x 1 x 1 x len_k
        k = k.repeat(1, 1, len_q, 1) # batch_size x 1 x len_q x len_k

        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3) # batch_size x 1 x len_q x 1
        q = q.repeat(1, 1, 1, len_k) # batch_size x 1 x len_q x len_k

        mask = k & q
        return mask

    ''''''''''''''''''''''''''
    '''2.2 Sub-Sequence mask'''
    ''''''''''''''''''''''''''
    def make_sequence_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask