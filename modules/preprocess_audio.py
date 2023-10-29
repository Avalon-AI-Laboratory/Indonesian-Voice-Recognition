from pydub import AudioSegment

AudioSegment.converter = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffprobe.exe"

import torch
import torchaudio.transforms as T_ta

from modules.base_function import conv2wav_torch, vad_torch, add_padding

class PreprocessAudio:
    def __init__(self, audio_file_directory, n_fft = 1024,
                 win_length = None, hop_length = 128, n_mels = 64):
        self.audio_file_directory = audio_file_directory
        self.mel_spectrogram_transform = T_ta.MelSpectrogram(sample_rate=8000,
                                                            n_fft=n_fft,
                                                            win_length=win_length,
                                                            hop_length=hop_length,
                                                            n_mels=n_mels,
                                                            mel_scale="htk")
        self.amplitude_to_db_transform = T_ta.AmplitudeToDB()

    def load_audio(self): # Make sure that the audio in mp3 format
        waves, sr = conv2wav_torch(self.audio_file_directory, 8000)
        signal_vad = vad_torch(waves, 1000, 0.012)
        mel_spectrogram = self.mel_spectrogram_transform(signal_vad.type(torch.float))
        log_mel_spectrogram = self.amplitude_to_db_transform(mel_spectrogram)
        log_mel_spectrogram = log_mel_spectrogram.permute(0, 2, 1)
        dataset = log_mel_spectrogram
        # print(dataset.shape)
        dataset = add_padding(dataset).detach() # add_padding returns NDArray
        # print("JOSSS")
        return dataset.repeat(4, 1, 1)