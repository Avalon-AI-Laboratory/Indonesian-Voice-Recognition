import torch
import torchaudio.transforms as T_ta
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pydub import AudioSegment
import os

from ops.base_functions import conv2wav_torch, vad_torch, add_padding, remove_outliers, binary_search, plot_scatter_before_after, conv_char2num

AudioSegment.converter = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffprobe.exe"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_MELS = 192

class ConvertWavToSpectogram:
    def __init__(self, n_fft = 1024, win_length = 400, hop_length = 200, n_mels = N_MELS):
        self.mel_spectrogram_transform = T_ta.MelSpectrogram(sample_rate=16000,
                                                            n_fft=n_fft,
                                                            win_length=win_length,
                                                            hop_length=hop_length,
                                                            n_mels=n_mels,
                                                            mel_scale="htk")
        self.amplitude_to_db_transform = T_ta.AmplitudeToDB()

    def transform(self, audio): # Make sure that the audio in mp3 format
        waves, _ = conv2wav_torch(audio, 16000)
        signal_vad = vad_torch(waves, 1000, 0.012) # Remove any moment of silence using VAD
        mel_spectrogram = self.mel_spectrogram_transform(torch.tensor(signal_vad).type(torch.float)) # Transform the cleaned signal to Mel Spectogram
        log_mel_spectrogram = self.amplitude_to_db_transform(mel_spectrogram) # Convert amplitude to decibel unit

        return log_mel_spectrogram


class AudioDataset(Dataset):
    def __init__(self, transcriptions, audio_dir):
        self.transcriptions = transcriptions.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.convert2spectogram = ConvertWavToSpectogram()

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        audio_dir = os.path.join(self.audio_dir, self.transcriptions['path'][idx])
        return self.load_audio(audio_dir), torch.tensor(conv_char2num(self.transcriptions['sentence'][idx])).detach().to(device)

    def load_audio(self, audiofile):
        # try:
        if (audiofile[-3:] == 'wav'):
            filename = f'{audiofile[:-4]}.wav'
            mp3file = AudioSegment.from_mp3(audiofile)
            mp3file.export(filename, format='wav')
            os.remove(audiofile)
            audiofile = filename
        
        log_mel_spectrogram = self.convert2spectogram.transform(audiofile).tolist()
        # except:
        #     print(f"Error dalam mengambil audio, kemungkinan audio ini tidak memiliki aktivitas suara")
        
        return torch.tensor(add_padding(log_mel_spectrogram, n_mels = N_MELS, max_padding = 728)).to(device)
