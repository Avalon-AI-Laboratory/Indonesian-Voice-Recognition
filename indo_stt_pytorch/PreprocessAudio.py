import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import math
import numpy as np
from pydub import AudioSegment

def conv2wav_torch(audio_dir, resample):
    mp3 = AudioSegment.from_mp3(audio_dir)
    wav = mp3.export(f'{audio_dir[:-4]}.wav', format="wav")
    waves, sr = torchaudio.load(f'{audio_dir[:-4]}.wav')
    waves = F.resample(waves, orig_freq=sr, new_freq=resample)
    return waves, resample

def vad_torch(waves, buff_size, threshold, display_info=False):
    mono_signal = waves[0].numpy()
    total_sig   = int(mono_signal.shape[0]/buff_size)
    signal      = []
    for i in range(total_sig):
        sig = mono_signal[i*buff_size:(i+1)*buff_size]
        rms = math.sqrt(np.square(sig).mean())
        if(rms > threshold):
            signal = np.append(signal,sig)
    signal = signal.astype('float')
    if (display_info):
        print("Number of total signal (signal_arr/buff_size):", total_sig)
        print("Signal data type:", signal.dtype)
        print(f"Signal shape: ({signal.shape})")
    return torch.tensor([signal])

class PreprocessAudio:
    def __init__(self, audio_dir, n_fft = 1024, #transcript_txt
                 win_length = None, hop_length = 128, n_mels = 64, n_mfcc = 64):
        self.audio_dir = audio_dir
        # self.transcript_txt = transcript_txt
        self.mfcc_transform = T.MFCC(sample_rate=8000,
                                     n_mfcc=n_mfcc,
                                     melkwargs={"n_fft": n_fft,
                                                "n_mels": n_mels,
                                                "hop_length": hop_length,
                                                "mel_scale": "htk",
                                               })
    def load_audio(self):
        print("Mounted audio directory at:", self.audio_dir)
        _dir = os.listdir(self.audio_dir)
        dataset = []
        for mp3 in _dir:
            try:
                if (mp3[-3:] == 'wav'):
                    continue
                mp3 = self.audio_dir + mp3
                waves, sr = conv2wav_torch(mp3, 8000)
                signal_vad = vad_torch(waves, 1000, 0.012)
                mfcc = self.mfcc_transform(signal_vad.type(torch.float))
                mfcc = mfcc.permute(0, 2, 1)
                # print(mfcc.shape)
                dataset.append(mfcc.numpy().tolist())
                os.unlink(mp3)
            except:
                print(f"Error di file {mp3}")
                return

        return dataset
