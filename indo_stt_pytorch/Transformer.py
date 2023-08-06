import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)
    
    def forward(self, x):
        maxlen = x.shape[-1]
        x = self.emb(x)
        positions = torch.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        
        return x + positions
        
class SpeechFeatureEmbedding(nn.Module):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.conv2 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(self.head_dim)
        
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, value)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.fc_out(out)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        
    def forward(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_att = MultiHeadAttention(embed_dim, num_heads)
        self.enc_att = MultiHeadAttention(embed_dim, num_heads)
        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )
    
    def causalAttentionMask(self, batch_size, n_dest, n_src, dtype):
        i = torch.arange(n_dest)[:, None]
        j = torch.arange(n_src)
        m = i >= j - n_src + n_dest
        mask = m.to(dtype)
        mask = mask.reshape(1, n_dest, n_src)
        mult = torch.cat([torch.tensor([batch_size], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32)])
        mult = mult.unsqueeze(0)
        return mask.expand(*mult)
    
    def forward(self, enc_out, target):
        input_shape = target.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causalAttentionMask(batch_size, seq_len, seq_len, target.dtype)
        target_att = self.self_att(target, target, attention_mask=causal_mask)

        
# MASIH BINGUNG DI MULTIHEADATTENTION (DETAIL: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
