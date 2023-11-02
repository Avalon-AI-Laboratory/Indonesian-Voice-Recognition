#!/bin/bash

mkdir audio_folder
mkdir audio_folder/train
mkdir audio_folder/valid
mkdir batch_data
mkdir common_voice_id

wget "drive.google.com/u/3/uc?id=1NfCGqWJOxYG0XOWDj0K4fD3mPB7RWHRR&export=download&confirm=yes"
tar -xf cv-corpus-15.0-2023-09-08-id.tar

mv cv-corpus-15.0-2023-09-08/id ./common_voice_id