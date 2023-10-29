import fire
import sounddevice as sd
import wavio
import numpy as np
import torch
from modules.preprocess_audio import PreprocessAudio
from modules.predict import predict
import warnings
import os
import streamlit as st

warnings.filterwarnings("ignore")

RATE = 44100
CHANNELS = 2
DTYPE = np.int32
SECONDS = 4

st.title("Audio Processing with Streamlit")

def start():
    audio_file_directory = record_audio()

    ################## PREPROCESS AUDIO AND RETURNS A TENSOR #####################
    data = PreprocessAudio(audio_file_directory).load_audio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data.to(device)
    ########################################################################################

    ################# PERFORM INFERENCE ########################################################
    prediction = predict(data)
    return prediction

def record_audio():
    st.write("Click the 'Record' button and speak for 4 seconds.")

    with st.empty():
        audio_data = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype=DTYPE)
        sd.wait()

    audio_file_directory = "./audio_container/input.wav"
    wavio.write(audio_file_directory, audio_data, RATE)
    st.write("Recording finished, inferencing...")
    return audio_file_directory

if st.button("Start Recording"):
    st.write("Recording...")
    prediction = start()
    with st.expander("Transkrip"):
        st.info(prediction)
