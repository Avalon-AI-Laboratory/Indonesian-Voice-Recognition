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

Idris :
* Need to reevaluate the data loader since the matrix sizes are still incorrect and need to be fixed. Additionally, the batch size also needs to be evaluated.
![Error](img/WhatsApp%20Image%202023-07-21%20at%2020.51.16.jpg).

## 26/07/2023
Idris : 
* Successfully filtered audio with low correlation between voice and transcript.
* Before :
  
![Before](img/LpYAAAAASUVORK5CYII.png)
* After :
  
![After](img/wOvEjtAVcBTHgAAAABJRU5ErkJggg.png)

Kevin :
* Updated main.ipynb, model.py, and PreprocessAudio.py
* Improving visualization and further audio cleaning

## 28/07/2023
Kevin ft. Idris:
* Trying implement DeepSpeech2 architecture on LJSpeech Dataset, returns about 15% word error rate

## 30/07/2023
Kevin ft. Idris
* Since DeepSpeech2 is a pretrained model and not our original, we want to build our model again using our modified transformer

## 1/08/2023
Kevin ft.  Idris
* Still discussing about Transformer architecture from "Attention is All You Need Paper"

## 3/08/2023
Kevin ft.  Idris
* Resolved data loading error caused by mismatched matrix sizes in the data loader.
* Experimented with different batch sizes to optimize model training performance.

## 5/08/2023
Kevin ft.  Idris
*Investigated and debugged the gradient exploding issue during model training.

## 7/08/2023
Kevin ft.  Idris
* Experimented with implementing a model architecture using TensorFlow.
* Decided to transform the TensorFlow architecture into a PyTorch architecture to align with the existing code.

## 9/08/2023
Kevin
* Finalized the adjustments to the model architecture to ensure compatibility with the existing PyTorch code.
* Conducted thorough testing to confirm the proper functionality and integrity of the updated architecture.

## 11/08/2023
Kevin
* Identified a critical issue in the ivr_project.ipynb code related to padding, realizing that padding with -1 was incorrect.
* Rectified the padding approach, making sure that it aligns with the correct requirements.

## 13/08/2023
Kevin
* Successfully resolved errors in the transformer model implementation.

## 15/08/2023
Idris
* Explored an alternative approach to audio preprocessing by applying audio enhancement techniques before filtering.

## 17/08/2023
Kevin
* Debugging to identified and addressed minor errors within the transformer

## 19/08/2023
Kevin
* Conducted a re-evaluation of the project's progress and current state.

Idris
* Attempted audio enhancement on the data to improve its quality before the filtering process.

## 21/08/2023
Kevin
* Discussed adjustments or refinements based on the evaluation to ensure project alignment with goals.

Idris
* Observed that the enhanced audio resulted in unclear and even worse quality compared to the original, leading to a decision to cancel this approach.
* Reverted to the previous audio preprocessing strategy due to the unfavorable results from the enhancement attempt.

## 23/08/2023
Kevin:
* Successfully fixed the forward propagation for transformer, conducting test for model training.
