## 18/07/2023
Kevin:
* Created main.ipynb and RNNT.py
* Created MFCC Feature Extraction functions for PyTorch module (including VAD to remove silence moments)
* Adjusted MFCC parameters to obtain better MFCC spectogram

## 20/07/2023
Kevin:
* Edited RNN_T.py
* Created PreprocessAudio.py as a module to load and preprocess audio dataset from a directory
* Still figuring out how RNN_T works to receive dynamic-sized tensor of MFCC (which are having different timesteps)

Idris:
* Normalize audio that has been processed with VAD
* Implementing audio enhancements to clarify the resulting sound

## 21/07/2023
Kevin:
* Built model:
![Model Architecture](https://github.com/Avalon-AI-Laboratory/Indonesian-Voice-Recognition/blob/9af33e8ee18c944bff26bd4d072c9998cf40915d/img/Screenshot%202023-07-21%20004511.png)
* Changed model CRNN with output consisting of a (1, 256, 28) matrix.
![Model_Architecture_Rev](https://github.com/Avalon-AI-Laboratory/Indonesian-Voice-Recognition/blob/6febd02fd7d74e7121f1984a23c6fc7920db2667/img/Screenshot%202023-07-21%20191832.png)

## 26/07/2023
Idris : 
* Successfully filtered audio with low correlation between voice and transcript.
