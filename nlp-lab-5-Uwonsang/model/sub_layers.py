import torch
from torch import nn
import math



class PostionalEncoding(nn.Module):

    def __init__(self, dim_model, max_seq_len, device):
        super(PostionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_seq_len, dim_model, device=device)  # seq(20), dim_model(512)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_seq_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # [seq(20), 1]

        _2i = torch.arange(0, dim_model, step=2, device=device).float() # _2i(256) -> (512/2i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim_model))) ## 0::2 -> 0번인덱스에서 2개찍 띄어가면서
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim_model))) ## 1::2 -> 1번인덱스에서 2개찍 띄어가면서

    def forward(self, x):

        batch_size, seq_len = x.size() # batch_size = 128, seq_len = 20

        return self.encoding[:seq_len, :]  # seq(20), dim_model(512)



''''''''''''''''''''''''''''''
'''5. embedding + Positional encoding '''
''''''''''''''''''''''''''''''

class Embedding_Layer(nn.Module):

    def __init__(self, num_token, dim_model, max_seq_len, device):
        super(Embedding_Layer, self).__init__()
        self.embedding = nn.Embedding(num_token, dim_model)
        self.positional_encoding = PostionalEncoding(dim_model, max_seq_len, device)

    def forward(self, x):
        emb = self.embedding(x)
        pos_emb = self.positional_encoding(x)
        return  emb + pos_emb

''''''''''''''''''''''''''''''
'''2. ScaleDotProductAttention'''
''''''''''''''''''''''''''''''
class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-9):

        batch_size, head, length, d_tensor = k.size() # batch_size, head, length, d_tensor
        k_t = k.transpose(2, 3)  # -> batch_size, head, d_tensor, length
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = self.softmax(score)
        v = score @ v

        return v, score

''''''''''''''''''''''''''''''
'''3. MultiHeadAttention'''
''''''''''''''''''''''''''''''
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.ScaleDot_attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # (q, k, v) -> batch, seq, model_dim(512)

        split_q, split_k, split_v = self.split(q), self.split(k), self.split(v) # (q, k, v) -> batch, head, seq, model_dim(64)
        out, attention = self.ScaleDot_attention(split_q, split_k, split_v, mask=mask)

        out = self.concat(out)
        final_out = self.out_linear(out)

        return final_out


    def split(self, tensor):

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):

        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


''''''''''''''''''''''''''''''
'''4. Feed Forward Layer'''
''''''''''''''''''''''''''''''
class FeedForward(nn.Module):

    def __init__(self, d_model, hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden)

    def forward(self, x, s_mask):
        attn_x = self.multi_head_attention(q=x, k=x, v=x, mask=s_mask)
        add_x = x + attn_x

        ffn_x = self.ffn(add_x)
        final_x = ffn_x + add_x

        return final_x

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, device):
        super().__init__()
        self.emb = Embedding_Layer(num_token=enc_voc_size, dim_model=d_model, max_seq_len=max_len, device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, s_mask)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden)

    def forward(self, dec, enc, t_mask, s_mask):

        attn1_x = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)
        add1_x = dec + attn1_x

        attn2_x = self.enc_dec_attention(q=add1_x, k=enc, v=enc, mask=s_mask)
        add2_x = add1_x + attn2_x

        ffn_x = self.ffn(add2_x)
        final_x = ffn_x + add2_x

        return final_x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, device):
        super().__init__()
        self.emb = Embedding_Layer(num_token=dec_voc_size, dim_model=d_model,
                                   max_seq_len=max_len, device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        emb_trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(emb_trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)
        return output
