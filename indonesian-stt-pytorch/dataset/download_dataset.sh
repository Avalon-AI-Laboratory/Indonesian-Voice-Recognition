#!/bin/bash

mkdir audio_folder
mkdir audio_folder/train
mkdir audio_folder/valid
mkdir batch_data

wget https://drive.google.com/dummy_link/cv-corpus-15.0-2023-09-08-id.tar
tar -xf cv-corpus-15.0-2023-09-08-id.tar

mv cv-corpus-15.0-2023-09-08/id ./common_voice_id
