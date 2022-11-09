import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
# from transformers.modeling_transfo_xl import PositionalEmbedding
from entmax import Entmax15, EntmaxBisect
from functools import partial

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]
        
        
        
class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
        return x
    
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model,device, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        
#         pe = pe.to(device)
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False
#         pe.requires_grad = False
    
    def forward(self, x):
        get_cuda_device = x.get_device()
#         print(get_cuda_device, self.d_model, x.size(), self.pe.size(), np.sqrt(self.d_model))
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
#         seq_len = x.size(1)
#         print(seq_len)
        x = x + self.pe.to(get_cuda_device)
#         print(x.size())
        return x
    

    
class SuperPositionalEmbedding(PositionalEmbedding):
    """
    Same as PositionalEmbedding in XLTransformer, BUT
    has a different handling of the batch dimension that avoids cumbersome dimension shuffling
    """

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.unsqueeze(0)
        if bsz is not None:
            pos_emb = pos_emb.expand(bsz, -1, -1)
        return pos_emb


class SuperPositionalBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings, BUT
    uses non-learnt (computed) positional embeddings
    """

    def __init__(self, vocab_size, hidden_size,hidden_dropout_prob=0.0):
        super(SuperPositionalBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = SuperPositionalEmbedding(hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids):
        # do word embedding first to determine its type (float or half)
        words_embeddings = self.word_embeddings(input_ids)

        # if position_ids or token_type_ids were not provided, used defaults
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=words_embeddings.dtype, device=words_embeddings.device)

        position_embeddings = self.position_embeddings(position_ids, input_ids.size(0))

        embeddings = words_embeddings + position_embeddings 
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, attn_func="softmax"):
        super().__init__()
        
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        
        Entmax = partial(EntmaxBisect, alpha=1.5, n_iter=30)
        attn_funcs = {"softmax": nn.Softmax,
                      "entmax15": Entmax15,
                      "entmax": Entmax}
        
        
        assert attn_func in attn_funcs, "Unknown attention function"
        self.transform = attn_funcs[attn_func](dim=-1)
        
        
        
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            scores = scores.masked_fill(mask == 0, -1e9)
#         scores = F.softmax(scores, dim=-1)

        # apply attention dropout and compute context vectors.
        attention = self.transform(scores)
        attention = self.dropout(attention)
    
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        sl = q.size(1)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, sl, self.h, self.d_k)
        q = self.q_linear(q).view(bs, sl, self.h, self.d_k)
        v = self.v_linear(v).view(bs, sl, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
    
class MultiHeadAttentionUnscaled(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
    
        scores = torch.matmul(q, k.transpose(-2, -1))  
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        sl = q.size(1)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, sl, self.h, self.d_k)
        q = self.q_linear(q).view(bs, sl, self.h, self.d_k)
        v = self.v_linear(v).view(bs, sl, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, attn_func="softmax"):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, attn_func=attn_func)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm_1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm_2(x)
        return x
    
class EncoderLayerHighWay(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        self.highway_att = Highway(2,   d_model)
        self.highway_ff = Highway(2,   d_model)
        
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm_1(x)
        x = self.highway_att(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm_2(x)
        x = self.highway_ff(x)
        return x
    
class TransformerEmbeddingHighWay(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, d_model, heads, layers, device, dropout=0.1):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,device, sequence_len)
        
        self.layers = nn.ModuleList([EncoderLayerHighWay(d_model, heads, dropout) for _ in range(layers)])
        self.norm = Norm(d_model)
        self.padding_id = 0
        self.embedding_dim = d_model
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pe(x)
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)
    

class TransformerCapsEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=50, dropout=0.1):
        super(TransformerCapsEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model)
        # # Implementation of Feedforward model
        #         # self.linear1 = Linear(d_model, dim_feedforward)
        #         # self.dropout = Dropout(dropout)
        #         # self.linear2 = Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.out_size = d_model
        self.feedforward_caps = CapsNet_Text(
            d_model,
            1,
            d_model,
            dim_caps=4,
            num_caps=8,
            num_compressed_caps=dim_feedforward,
        )
        self.scaling_param = nn.Parameter(torch.tensor([1e-8]))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        
        x = x + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)
        B, N, d_c = x.size()
        x2 = x.view(-1, 1, self.d_model)
        _, x2 = self.feedforward_caps(x2)
#         print(x2.size(), B, N, self.out_size)
#         print(self.dropout2(x2.squeeze(2)).view(B, N, self.out_size).size())
        x = x + self.dropout2(x2.squeeze(2)).view(B, N, self.out_size)
         
        x = self.norm2(x)
        
        
#         src2 = self.self_attn(
#             src, src, src, src_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         B, N, d_c = src.size()
#         src = src.view(-1, 1, self.d_model)
#         _, src2 = self.feedforward_caps(src)
#         src = self.dropout2(src2.squeeze(2)).view(B, N, self.out_size)

#         src = torch.log(src + self.scaling_param / (1 - src + self.scaling_param))
#         src = self.norm2(src)

        # src = src * torch.pow(10, self.scaling_param)

        return x



    
    
class TransformerEmbedding(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, d_model, heads, layers, device, dropout=0.1, attn_func="softmax"):
        
        super().__init__()
        
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pe = PositionalEncoder(d_model,device, sequence_len)
        
        self.embedding = SuperPositionalBertEmbeddings(vocab_size, d_model)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout, attn_func=attn_func) for _ in range(layers)])
        self.norm = Norm(d_model)
        self.padding_id = 0
        self.embedding_dim = d_model
        
    def forward(self, x, mask):
        x = self.embedding(x)
#         x = self.pe(x)
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)

class TransformerCapsEmbedding(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, d_model, heads, layers, device, dropout=0.1):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,device, sequence_len)
        
        self.layers = nn.ModuleList([TransformerCapsEncoderLayer(d_model, heads, dim_feedforward=50,dropout=dropout) for _ in range(layers)])
        self.norm = Norm(d_model)
        self.padding_id = 0
        self.embedding_dim = d_model
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pe(x)
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)

    
class EncoderLayerUnscaled(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttentionUnscaled(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm_1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm_2(x)
        return x
    

    
class EncoderLayerUnscaledHighWay(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttentionUnscaled(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)
        self.highway_att = Highway(2,   d_model)
        self.highway_ff = Highway(2,   d_model)
        
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm_1(x)
        x = self.highway_att(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm_2(x)
        x = self.highway_ff(x)
        return x
    
class TransformerEmbeddingUnscaled(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, d_model, heads, layers, device, dropout=0.1):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,device, sequence_len)
        
        self.layers = nn.ModuleList([EncoderLayerUnscaled(d_model, heads, dropout) for _ in range(layers)])
        self.norm = Norm(d_model)
        self.padding_id = 0
        self.embedding_dim = d_model
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pe(x)
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)
    
class TransformerEmbeddingUnscaledHighWay(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, d_model, heads, layers, device, dropout=0.1):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model,device, sequence_len)
        
        self.layers = nn.ModuleList([EncoderLayerHighWay(d_model, heads, dropout) for _ in range(layers)])
        self.norm = Norm(d_model)
        self.padding_id = 0
        self.embedding_dim = d_model
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pe(x)
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)
