import torch
import torch.nn as nn

# from linformer_pytorch import LinformerLM, LinformerEncDec
# from performer_pytorch import PerformerLM, PerformerEncDec
# from reformer_pytorch import ReformerLM, ReformerEncDec
from transformer_pytorch import TransformerEmbedding,TransformerEmbeddingUnscaled,TransformerEmbeddingHighWay,TransformerEmbeddingUnscaledHighWay,TransformerCapsEmbedding
import torch.nn.functional as F

class GenearalTransformer(nn.Module):

    def __init__(self, model, vocab_size, sequence_len, d_model, heads, layers, device, attn_func="softmax"):
        super().__init__()

        if model not in ['transformer', 'linformer', 'performer', 'reformer', 'transformerU', 'transformerH', 'transformerUH', 'TransformerCaps']:
            raise Exception(f'Unknown model type {model}. Model type must be one of transformer, linformer, performer, reformer')

        self.model_type = model

        if model == 'transformer':
            self.encoder = TransformerEmbedding(vocab_size, sequence_len, d_model, heads, layers, device, attn_func=attn_func)
        elif model == 'linformer':
            self.encoder = LinformerLM(num_tokens=vocab_size, input_size=sequence_len, channels=d_model, 
                                       nhead=heads, depth=layers, return_emb=True)
        elif model == 'performer':
            self.encoder = PerformerLM(num_tokens=vocab_size, max_seq_len=sequence_len, dim=d_model, 
                                       heads=heads, depth=layers)
        elif model == 'reformer':
            self.encoder = ReformerLM(num_tokens=vocab_size, max_seq_len=sequence_len, dim=d_model, 
                                       heads=heads, depth=layers, return_embeddings=True)
        elif model == 'transformerU':
            self.encoder = TransformerEmbeddingUnscaled(vocab_size, sequence_len, d_model, heads, layers, device)
        elif model == 'transformerH':
            self.encoder = TransformerEmbeddingHighWay(vocab_size, sequence_len, d_model, heads, layers, device)
        elif model == 'transformerUH':
            self.encoder = TransformerEmbeddingUnscaledHighWay(vocab_size, sequence_len, d_model, heads, layers, device)
        elif model == 'TransformerCaps':
            self.encoder = TransformerCapsEmbedding(vocab_size, sequence_len, d_model, heads, layers, device)

        else:
            pass

    def forward(self, X, mask):
        if self.model_type == 'transformer':
            return self.encoder(X, mask)
        elif self.model_type == 'performer':
            return self.encoder(X, return_encodings=True)
        elif self.model_type == 'linformer':
            return self.encoder(X, input_mask=mask, embedding_mask=mask)
        elif self.model_type == 'transformerU':
            return self.encoder(X, mask)
        elif self.model_type == 'transformerH':
            return self.encoder(X, mask)
        elif self.model_type == 'transformerUH':
            return self.encoder(X, mask)
        elif self.model_type == 'TransformerCaps':
            return self.encoder(X, mask)
        else:
            return self.encoder(X, input_mask=mask, context_mask=mask)
        
# class GenearalTransformerEncDec(nn.Module):

#     def __init__(self, model, vocab_size_enc, sequence_len_enc,vocab_size_dec, sequence_len_dec, d_model, heads, layers):
#         super().__init__()

#         if model not in [ 'linformer', 'performer', 'reformer']:
#             raise Exception(f'Unknown model type {model}. Model type must be one of transformer, linformer, performer, reformer')

#         self.model_type = model

#         if model == 'transformer':
#             print('unimplemented')
#            # self.encoder = TransformerEmbedding(vocab_size, sequence_len, d_model, heads, layers)
#         elif model == 'linformer':
#             self.encoder = LinformerEncDec(enc_num_tokens=vocab_size_enc, enc_input_size=sequence_len_enc, 
#                                            dec_num_tokens=vocab_size_dec,dec_input_size = sequence_len_dec,
#                                            enc_channels=16,dec_channels=16,
#                                        nhead=heads, depth=layers, return_emb=True)
#         elif model == 'performer':
#             self.encoder = PerformerEncDec(enc_num_tokens=vocab_size_enc, enc_max_seq_len=sequence_len_enc, 
#                                            dec_num_tokens=vocab_size_dec,dec_max_seq_len = sequence_len_dec,
#                                            dim=d_model, 
#                                        heads=heads, depth=layers)
#         elif model == 'reformer':
#             self.encoder = ReformerEncDec(enc_num_tokens=vocab_size_enc, enc_max_seq_len=sequence_len_enc, 
#                                            dec_num_tokens=vocab_size_dec,dec_max_seq_len = sequence_len_dec,
#                                       dim=d_model, 
#                                        heads=heads, depth=layers, return_embeddings=True)
#         else:
#             pass

#     def forward(self, enc_X, enc_mask, dec_X, dec_mask):
#         if self.model_type == 'transformer':
#             return self.encoder(X, mask)
#         elif self.model_type == 'performer':
#             return self.encoder(X, return_encodings=True)
#         elif self.model_type == 'linformer':
#             return self.encoder(X, input_mask=mask, embedding_mask=mask)
#         else:
#             return self.encoder(X, input_mask=mask, context_mask=mask)
        
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
    
    
    
    
    
    
    
    
    

            