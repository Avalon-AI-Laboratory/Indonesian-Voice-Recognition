import math
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_ta
import os
import matplotlib.pyplot as plt

TARGET_MAXLEN = 257

''' CHARACTER-TO-NUMERIC UTILITIES '''
alphabets = ['', ' ', '<', '>'] + [chr(i + 96) for i in range(1, 27)]
char2num_dict, num2char_dict = {}, {}

for index, chars in enumerate(alphabets):
    char2num_dict[chars] = index
    num2char_dict[index] = chars

def conv_char2num(label, maxlen=TARGET_MAXLEN):
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


''' AUDIO MANIPULATION UTILITIES '''

def conv2wav_torch(file_audio, resample):
    waves, sr = torchaudio.load(file_audio)
    waves = F_ta.resample(waves, orig_freq=sr, new_freq=resample)
    os.remove(file_audio)  # Remove the temporary WAV file after processing
    return waves, resample

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

    return signal

def add_padding(audio_tensor, n_mels, max_padding):
    height, width = np.array(audio_tensor).shape[-2], np.array(audio_tensor).shape[-1]
    
    padded_mel_spec = np.zeros([n_mels, max_padding])
    padded_mel_spec[:height, :width] = audio_tensor
    return padded_mel_spec

''' REMOVE DEFECTIVE DATA UTILITIES '''

def calculate_distance(x, y, m, c):
    return abs(y - (m * x + c)) / np.sqrt(m**2 + 1)

def remove_outliers(df, P):
    m, c = np.polyfit(df['transcript_len'], df['spectogram_len'], 1)
    df['distance'] = calculate_distance(df['transcript_len'], df['spectogram_len'], m, c)
    return df[df['distance'] <= P].drop(columns=['distance'])

def binary_search(arr, x):
    low, mid, high = 0, 0, len(arr) - 1
    
    while low <= high:
        mid = (high + low) // 2
        if arr[mid] < x:
            low = mid + 1
        
        elif arr[mid] > x:
            high = mid - 1
        
        else:
            return mid
    
    return -1


''' VISUALIZATION UTILITY '''

def plot_scatter_before_after(df_before, df_after, threshold):
    m, c = np.polyfit(df_before['transcript_len'], df_before['spectogram_len'], 1)
    plt.scatter(df_before['transcript_len'], df_before['spectogram_len'], color='orange', label='Before Cleaning')
    plt.scatter(df_after['transcript_len'], df_after['spectogram_len'], color='blue', label='After Cleaning')
    
    x = np.array(df_before['transcript_len'])
    y = m * x + c
    plt.plot(x, y, '-r', label='Regression Line')
    
    y_upper = y + threshold * np.sqrt(1 + m**2)
    y_lower = y - threshold * np.sqrt(1 + m**2)
    
    plt.plot(x, y_upper, '--g', label=f'Threshold Line (+{threshold})')
    plt.plot(x, y_lower, '--g', label=f'Threshold Line (-{threshold})')
    
    plt.xlabel('Transcript Length')
    plt.ylabel('Spectogram Length')
    plt.legend()
    plt.title('Scatter Plot Before and After Removing Defective Data')
    plt.grid()
    plt.show()