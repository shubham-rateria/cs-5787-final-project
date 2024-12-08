import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    # print(f"{q.shape}, {k.mT.shape}")
    attn_logits = torch.matmul(q, k.mT)
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttn(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cls_token = "[CLS]"
        
        # x -> b, t, input_dim | t=seq_len
        # qkv_proj = h * d_k *3, output = b, t, h * d_k * 3
        self.qkv_proj = nn.Linear(input_dim, self.num_heads * self.embed_dim * 3)
        self.o_proj = nn.Linear(self.num_heads * self.embed_dim, self.embed_dim)
    
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, x, mask=None, return_attention=True):
        # print(f"\n multihead_attn - x.shape: {x.shape} \n")
        b, t, _ = x.shape
        # qkv -> b, t, h * dim * 3
        qkv = self.qkv_proj(x)
        # print(f"qkv shape: {qkv.shape}")
        qkv = qkv.reshape(b, t, self.num_heads * self.embed_dim, 3)
        # print(f"Reshaped qkv shape: {qkv.shape}")
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        
        values, attention = scaled_dot_product(q, k, v)
        # print(f"Values shape: {values.shape}, Attention shape: {attention.shape}")
        
        # values = values.reshape(b, t, self.num_heads * self.embed_dim)
        # print(f"Reshaped values shape: {values.shape}")
        output = self.o_proj(values)
        # print(f"Output shape: {output.shape}")
        
        if return_attention:
            return output, attention
        else:
            return output
        
class Encoder(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttn(input_dim, input_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask, return_attention=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        x = self.mlp(x)
        x = x + self.dropout(x)
        x = self.norm2(x)
        
        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([Encoder(**block_args) for _ in  range(num_layers)])
        
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

if __name__ == "__main__":
    multihead_attn = MultiHeadAttn(input_dim=512, embed_dim=64, num_heads=1)
    encoder = Encoder(512, 8, 128)
    x = torch.randn(1, 10, 512)
    y = multihead_attn(x)
    y = encoder(x)
    print(f"shape: {y.shape}")
    # print(f"Output shape: {output.shape}, Attention shape: {attention.shape}")

