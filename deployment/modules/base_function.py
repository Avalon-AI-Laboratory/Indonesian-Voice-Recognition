import math
import numpy as np
from pydub import AudioSegment

import torch
import torchaudio
import torchaudio.functional as F_ta
import os

alphabets = ['', ' ', '<', '>'] + [chr(i + 96) for i in range(1, 27)]
char2num_dict, num2char_dict = {}, {}

for index, chars in enumerate(alphabets):
    char2num_dict[chars] = index
    num2char_dict[index] = chars


def conv2wav_torch(file_audio, resample):
    waves, sr = torchaudio.load(file_audio)
    waves = F_ta.resample(waves, orig_freq=sr, new_freq=resample)
    os.remove(file_audio)  # Remove sthe temporary WAV file after processing
    return waves, resample

def conv_char2num(label, maxlen=257):
    label = label[:maxlen].lower()
    label_enc = [char2num_dict['<']]
    padding_len = maxlen - len(label)
    for i in label:
        label_enc.append(char2num_dict[i])
    label_enc.append(char2num_dict['>'])
    return np.array(label_enc + [0] * padding_len)

def conv_num2char(num):
    txt = ""
    for i in num:
        if i == 0:
            break
        else:
            txt += num2char_dict[i]
    
    return txt

def vad_torch(waves, buffer_size, threshold, display_info = False):
    mono_signal = waves[0].numpy()
    total_signal = int(mono_signal.shape[0] / buffer_size)
    signal = np.array([])
    for i in range(total_signal):
        sig = mono_signal[i * buffer_size : (i + 1) * buffer_size]
        rms = math.sqrt(np.square(sig).mean())
        if (rms > threshold):
            signal = np.append(signal, sig)
    
    signal = signal.astype('float')

    if display_info:
        print("Number of total signal (signal_arr/buff_size):", total_signal)
        print("Signal data type:", signal.dtype)
        print(f"Signal shape: ({signal.shape})")

    return torch.tensor([signal])

def add_padding(audio_tensor, n_mels=64, max_padding=524):
    height, width = np.array(audio_tensor).shape[1], np.array(audio_tensor).shape[2]
    
    padded_mel_spec = np.zeros([max_padding, n_mels])
    padded_mel_spec[:height, :width] = audio_tensor[0]
    return torch.tensor([padded_mel_spec])
