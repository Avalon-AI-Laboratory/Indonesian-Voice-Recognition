import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        error_const = torch.erf(x / math.sqrt(2.0))
        x = x * 0.5 * (1.0 + error_const(x))
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=30, num_hid=64):
        super().__init__()
        self.num_vocab = num_vocab
        self.num_hid = num_hid

    def forward(self, x):
        maxlen = x.shape[-1]
        pos = torch.arange(0, maxlen, 1)
        emb = nn.Embedding(num_vocab, num_hid)(x)
        pos_emb = nn.Embedding(maxlen, num_hid)(x)
        return emb + pos_emb
        
class SpeechFeatureEmbedding(nn.Module):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.conv2 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.gelu = GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            GELU(),
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
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            GELU(),
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
        target_att = self.self_att(target, target, target, att_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

class Transformer(nn.Module):
    def __init__(
        self,
        num_hid = 64,
        num_head = 2,
        num_feed_forward = 128,
        source_maxlen = 100,
        target_maxlen = 100,
        num_layers_enc = 4,
        num_layers_dec = 1,
        num_classes = 10
    ):
        super().__init__()
        self.loss_metric = nn.MSELoss()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        
        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid)
        
        self.encoder = nn.Sequential(
            self.enc_input,
            *[TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_enc)]
        )
        
        for i in range(num_layers_dec):
            self.add_module(
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = nn.Linear(num_hid, num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            dec_layer = getattr(self, f"dec_layer_{i}")
            y = dec_layer(enc_out, y)
        return y

    def forward(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)
