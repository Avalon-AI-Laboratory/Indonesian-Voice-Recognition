import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ops.openai_gelu import GELU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextEmbedding(nn.Module):
    def __init__(self, num_vocab=30, num_hid=192, maxlen=257):  # If maxlen changed, change the attn_mask dim as well
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
        return pos_encoding.to(device)

    def forward(self, x):
        emb = self.emb(x.to(device))
        out = emb + self.pos_encoding[:emb.size(0), :]
        return out

class AudioFeaturesExtractionModule(nn.Module):
    def __init__(self, num_hid=192):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_hid, out_channels=num_hid*2, kernel_size=16, stride=2, padding=5)
        self.conv2 = nn.Conv1d(in_channels=num_hid*2, out_channels=num_hid*2, kernel_size=16, stride=2, padding=5)
        self.conv3 = nn.Conv1d(in_channels=num_hid*2, out_channels=num_hid, kernel_size=16, stride=2, padding=5)
        self.gelu = GELU()

    def forward(self, x):
        '''
        Saat menerima input matriks spektogram yang memiliki dimensi (batch_size, length, n_mels),
        modul ini harus melakukan transpose sehingga dimensinya menjadi (batch_size, n_mels, length).
        Sebagai contoh, misalkan sebuah matriks A berbentuk:
        [[[1, 2, 3],
          [4, 5, 6]],

         [[2, 4, 6],
          [2, 3, 4]]]
        Ini merupakan matriks input dengan batch_size = 2, channels = 2, length = 3, sehingga ukurannya (2, 2, 3).
        Konvolusi 1D akan berlangsung pada vektor-vektor di dalamnya seperti [1, 2, 3] dan [2, 4, 6]. Yang kita ingin
        konvolusikan adalah vektor setiap filter mel beserta panjangnya, dengan kata lain, jika kita membiarkan inputnya
        berdimensi (batch_size, length, n_mels), maka konvolusi akan dilakukan sepanjang frekuensi. Kita tidak
        menginginkan ini karena agar memahami fitur dari bagaimana karakter dalam suatu kata dilafalkan, model perlu
        memahami bagaimana pola sebuah mel bank berubah sepanjang waktu. Maka dari itu, konvolusi harus dilakukan pada
        waktu (length), sehingga dimensi yang harus diterima adalah (batch_size, n_mels, length).
        '''
        x = self.conv1(x) # (batch_size, length, n_mels) --> (batch_size, n_mels, length)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        return x.permute(0, 2, 1) # kembali ke bentuk awal
