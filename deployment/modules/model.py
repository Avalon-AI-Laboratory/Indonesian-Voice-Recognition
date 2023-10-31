import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        error_const = torch.erf(x / math.sqrt(2.0))
        x = x * 0.5 * (1.0 + error_const)
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=30, num_hid=64, maxlen=257):  # If maxlen changed, change the attn_mask dim as well
        super().__init__()
        self.maxlen = maxlen
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_encoding = self.create_pos_encoding(maxlen, num_hid)

    def create_pos_encoding(self, maxlen, num_hid):
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_hid, 2).float() * (-math.log(10000.0) / num_hid))
        pos_encoding = torch.zeros(maxlen, num_hid)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        return pos_encoding.to("cuda:0")

    def forward(self, x):
        emb = self.emb(x.to("cuda:0"))
        out = emb + self.pos_encoding[:emb.size(0), :]
        return out

class SpeechFeatureEmbedding(nn.Module):
    def __init__(self, num_hid=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.conv2 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid, kernel_size=11, stride=2, padding=5)
        self.gelu = GELU()

    def forward(self, x):
        x = self.conv1(x.permute(0, 2, 1))
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        return x.permute(0, 2, 1)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
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
        attn_output = self.att(inputs, inputs, inputs)[0]
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out = self.layernorm2(out1 + ffn_output)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            GELU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )

    def causalAttentionMask(self, size, dtype=float('-inf'), device='cpu'):
        return torch.triu(torch.full((size, size), dtype, device=device), diagonal=1)

    def forward(self, enc_out, target):
        causal_mask = self.causalAttentionMask(size=target.shape[1], dtype=float('-inf'), device=target.device)
        target_att = self.self_att(target, target, target, attn_mask=causal_mask)[0]
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out, enc_out)[0]
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
        num_layers_dec = 4,
        num_classes = 30
    ):
        super().__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, num_hid=num_hid)

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

    def decoder(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            dec_layer = getattr(self, f"dec_layer_{i}")
            y = dec_layer(enc_out, y)
        return y

    def forward(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decoder(x, target)
        return self.classifier(y)

    def generate(self, source, start_token_idx, max_len=257):
        batch_sz = source.shape[0]
        enc = self.encoder(source)
        dec_input = torch.ones(batch_sz, 1).to(torch.int32) * start_token_idx
        dec_input = dec_input.to(device)
        dec_logits = []
        for i in range(max_len - 1):
            dec_out = self.decoder(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = torch.argmax(logits, dim=-1).to(torch.int32)
            last_logit = logits[:, -1].unsqueeze(-1)
            dec_logits.append(last_logit)
            dec_input = torch.cat((dec_input, last_logit), dim=-1)
        return dec_input