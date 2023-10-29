import torch
from modules.model import Transformer
import sys
from modules.base_function import conv_num2char
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./config/transformer_model.pth').to(device)

def predict(audio_tensor, target_maxlen=257, start_token_idx=2, end_token_idx=3):
    audio_tensor = audio_tensor.float().to(device)
    prediction = model.generate(audio_tensor, start_token_idx)[0]
    prediction = conv_num2char(prediction.detach().cpu().numpy())

    matches = re.findall(r'[a-zA-Z\s]+', prediction)
    cleaned_text = matches[0] if matches else None
    return f"{cleaned_text}"