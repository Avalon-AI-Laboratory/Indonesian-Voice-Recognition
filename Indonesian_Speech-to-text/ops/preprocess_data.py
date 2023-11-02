import torch
import torchaudio.transforms as T_ta
import numpy as np
import pandas as pd
import os
from ops.base_functions import conv2wav_torch, vad_torch, add_padding, remove_outliers, binary_search, plot_scatter_before_after, conv_char2num
from pydub import AudioSegment

AudioSegment.converter = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffmpeg = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = "C:\\ProgramData\\chocolatey\\lib\\ffmpeg-full\\tools\\ffmpeg\\bin\\ffprobe.exe"

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

class RemoveDefectData:
    def __init__(self, log_mel_spectograms, transcriptions, hyperplane_threshold):
        self.log_mel_spectograms = log_mel_spectograms
        self.transcriptions = transcriptions
        self.hyperplane_threshold = hyperplane_threshold
    
    def apply(self):
        spectogram_len_list, transcription_len_list = [], []

        for i in range(len(self.log_mel_spectograms)):
            spectogram_len_list.append(np.array(self.log_mel_spectograms[i][1]).shape[0])
        
        for text in self.transcriptions['sentence']:
            transcription_len_list.append(len(text))

        pairs_dataframe = pd.DataFrame({'spectogram_len': spectogram_len_list, 
                                         'transcript_len': transcription_len_list})

        cleaned_pairs_dataframe = remove_outliers(pairs_dataframe, self.hyperplane_threshold)
        cleaned_index = pd.Series(cleaned_pairs_dataframe.index.tolist())
        default_index = pd.Series(self.transcriptions.index.tolist())
        id_to_drop = default_index[~default_index.isin(cleaned_index)].tolist()

        cleaned_transcripts = self.transcriptions.loc[cleaned_index].reset_index(drop = True)
        cleaned_spectograms = []

        plot_scatter_before_after(pairs_dataframe, cleaned_pairs_dataframe, self.hyperplane_threshold)

        for i in range(len(self.log_mel_spectograms)):
            pos = binary_search(id_to_drop, i)
            if pos != -1:
                continue

            cleaned_spectograms.append(self.log_mel_spectograms[i])
        
        return cleaned_spectograms, cleaned_transcripts, cleaned_pairs_dataframe

class PreprocessData:
    def __init__(self, audio_dir, transcription_df, hyperplane_threshold, return_metadata = True):
        self.audio_dir = audio_dir
        self.transcription_df = transcription_df
        self.hyperplane_threshold = hyperplane_threshold
        self.return_metadata = return_metadata
        self.convert2spectogram = ConvertWavToSpectogram()

    def load(self):
        counter = 0
        print("Mounted audio directory at: ", self.audio_dir)
        
        spectograms = []
        list_file_error = []
        list_index_error = []
        _dir = self.transcription_df['path']
        for audio in _dir:
            audiofile = self.audio_dir + audio
            try:
                if (audiofile[-3:] == 'wav'):
                    filename = f'{audiofile[:-4]}.wav'
                    mp3file = AudioSegment.from_mp3(audiofile)
                    mp3file.export(filename, format='wav')
                    os.remove(audiofile)
                    audiofile = filename
                
                log_mel_spectrogram = self.convert2spectogram.transform(audiofile)
                spectograms.append(log_mel_spectrogram.tolist())
                counter += 1
            except:
                print(f"Counter on {counter}, error on {audio}")
                list_file_error.append(audio)
                list_index_error.append(counter)
                self.transcription_df.drop(counter, inplace = True)
                counter += 1
                continue
        
        self.transcription_df = self.transcription_df.reset_index(drop = True)
        remove_defect_data = RemoveDefectData(spectograms, self.transcription_df, self.hyperplane_threshold)
        cleaned_spectograms, cleaned_transcripts, len_audio_transcript = remove_defect_data.apply()
        for i in range(len(cleaned_spectograms)):
             # add_padding returns NDArray
            cleaned_spectograms[i] = add_padding(cleaned_spectograms[i], n_mels = N_MELS, max_padding = np.max(len_audio_transcript['spectogram_len'])).tolist()
        
        cleaned_spectograms = torch.tensor(cleaned_spectograms) # Konversi ke torch tensor

        if not self.return_metadata:
            numeric_transcriptions = []
            for text in cleaned_transcripts['sentence']:
                try:
                    numeric_transcriptions.append(conv_char2num(text))
                except:
                    print("Proses tidak berhasil. Cek lagi preprocessing pada tanda baca atau elemen lain.")
            numeric_transcriptions = torch.tensor(numeric_transcriptions).detach()

            return cleaned_spectograms, numeric_transcriptions
        else:
            return cleaned_spectograms, cleaned_transcripts