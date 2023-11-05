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
* Successfully train the model. Building functions for displaying outputs (inference) and hyperparameter tuning.

## 25/08/2023
Kevin & Idris:
* Successfully implement backpropagation using cross entropy loss

## 27/08/2023
Kevin & Idris:
* No significant progress

## 29/08/2023
Kevin:
* The loss is observed and it returns relatively high loss score using cross entropy loss (about 1.278). Our hypothesize is that the data is insufficient. Thus we gather more data (about 40000).

![image](https://github.com/Avalon-AI-Laboratory/Indonesian-Voice-Recognition/assets/92637327/ecc04ed0-a10f-494f-ad08-cc025a665a50)
Idris:
* Preprocessed the data and fixed some parts of the preprocessing module (conv2wav_torch)

## 31/08/2023
Kevin & Idris:
* No significant progress

## 02/09/2023
Kevin & Idris:
* Stuck on computation memory when trying to preprocess 40000 data, looking for a way to optimize the code.

## 04/09/2023
Kevin & Idris:
* Still facing the same problem as before.

## 06/09/2023
Kevin & Idris:
* Tries to preprocess the audio using wavelet analysis and deeper preprocessing, such as dropping non-indonesian vocabulary. Trying to align every MFCC on the same frequency bandwidth. It doesn't give any significant results

## 08/09/2023
Idris:
* Requesting Google Colab Pro for solving computation memory problems. As a result, we successfully preprocessed 10000 MFCC data
Kevin:
* Initial result:

![image](https://github.com/Avalon-AI-Laboratory/Indonesian-Voice-Recognition/assets/92637327/b00ac8ae-bc29-4d50-b7d2-7efcad5f3aa4)
* Trying to fix the vanishing first character problem and solved it.
## 10/09/2023
Idris and Kevin:
* Training the data, did not give any significant results. Tried hyperparameter tuning such as changing feed-forward network layers count, encoder and decoder layers count, etc.

## 12/09/2023
* No significant progress

## 14/09/2023
* No significant progress

## 16/09/2023
Idris:
* Encountering memory overload when trying to preprocess 40000 data through Google Colab Pro

![image](https://github.com/Avalon-AI-Laboratory/Indonesian-Voice-Recognition/assets/92637327/e73eafe6-7252-46c2-9e72-ed0d2a40c1e5)
Kevin:
* Trying to implement Multiprocessing for preprocessing audio data. As a result, the device experienced system crash due to memory overleak

![image](https://github.com/Avalon-AI-Laboratory/Indonesian-Voice-Recognition/assets/92637327/e8a1176a-fb99-46d3-9ce8-f0be5731e636)

## 18/09/2023
Kevin & Idris:
* Continued to explore various techniques to optimize data preprocessing and model training.

## 20/09/2023
* No significant progress

## 22/09/2023
* No significant progress

## 24/09/2023
Kevin & Idris:
* Explored different approaches to handling memory overleak.
* Improved the efficiency of data preprocessing by optimizing code.

## 25/09/2023
Kevin:
* Conducted extensive testing and evaluation of the model's performance, identifying areas for improvement.

## 26/09/2023
Kevin:
* Focused on fine-tuning the model architecture.

## 28/09/2023
Kevin:
* Developing a evaluation framework to assess the model's performance, including metrics such as word error rate (WER) and accuracy.

Idris:
* Worked on enhancing the quality of the audio data further.

## 29/09/2023
Kevin:
* Continued to fine-tune the model based on the evaluation results, with a focus on reducing WER and improving overall accuracy.

## 03/10/2023
Kevin :
* Experimenting with different learning rates and batch sizes.

Idris:
* Exploring the implementation of a dynamic learning rate scheduler to improve model training.

## 05/10/2023
Idris:
* Optimizing audio preprocessing to make better eliminate silent segments.

Kevin:
* Conducted additional training runs with variations in batch sizes and learning rates to find the optimal hyperparameters.

## 07/10/2023
Kevin & Idris:
* Implemented a learning rate scheduler that has proven effective in enhancing model convergence.
* Addressed the issue of stagnation in the average loss value (avg_loss) by successfully reducing it to 0.8.

## 08/10/2023
Kevin:
* Undertook additional model testing with a specific focus on improving transcription accuracy.
* Achieved an improved transcription result for the phrase 'ibu sedang tidak ada di rumah,' with the prediction 'tu sadang tidak ada ai muma,' which indicates better word pattern recognition and alignment

## 10/10/2023
Kevin & idris:
* Achieved significant improvement in transcription accuracy, primarily attributed to a key revelation in the choice of the loss function. Transitioned from using Binary Cross-Entropy (BCE) to Connectionist Temporal Classification (CTC) loss function.

## 13/10/2023
Idris:
* Implemented a data management strategy to address memory overleak issues during audio preprocessing.
* Divided the dataset into batch files in the .pt format, resulting in the segmentation of the 40,000 data points into four equal parts, each containing 10,000 data points.
* This partitioning approach aims to optimize memory usage and streamline the preprocessing pipeline, ensuring more efficient data processing and model training.

![image](img/1249-1613-6913.png)

## 14/10/2023
Idris:
* Attempted model training using the entire 40,000 data points in a single batch, but encountered persistent memory limitations.
* Experienced recurring kernel crashes, even when utilizing a supercomputer for processing.

![image](img/image.png)

## 24/10/2023
Kevin & Idris:
* Optimizing memory utilization to enhance the overall performance of the project.
* Debugging significant memory consumption issue.

## 25/10/2023
Kevin:
* Explored a new approach in the Indonesian Voice Recognition project by transitioning from the use of Mel-frequency cepstral coefficients (MFCC) to Mel spectrograms for audio feature extraction.

## 27/10/2023
Kevin:
* Following the adoption of Mel spectrograms for audio feature extraction, observed a notable improvement in training performance.

## 28/10/2023
Kevin & Idris:
* Prepared the best-performing model and the necessary code for deployment in a Streamlit.

## 29/10/2023
Kevin & Idris:
* Successfully completed the deployment.

# 30/10/2023
Kevin & Idris:
* After multiple epochs of training, the model exhibited increasing signs of overfitting.
* Suspected issues within the model architecture that required reevaluation.

# 01/11/2023
Idris:
* Trying to re-evaluate models and reading existing references (https://its.id/m/SpeechRecognition)
Kevin:
* Trying to solve high computational memory problems by reading references and found that the problem can be solved by implementing batch-on-the-fly technique

# 03/11/2023
Kevin:
* Identified the cause of high computational memory problem. Until this time, the method we implemented to create a Dataset instance is by storing every preprocessed dataset in the main memory. This will cause memory overleak since we were trying to store an enormous-sized variable in memory. This problem is solved by implementing torch.utils.data.Dataset whereareas this class will fetch a data (audio file) directly from the audio folder, preprocess the data inside the class, and returns a spectogram matrix and the numerical representation of transcription. As a result, the training was successfully tested on 8 GB NVIDIA RAM using 25684 data (which was not possible before).
```python
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
        try:
            if (audiofile[-3:] == 'wav'):
                filename = f'{audiofile[:-4]}.wav'
                mp3file = AudioSegment.from_mp3(audiofile)
                mp3file.export(filename, format='wav')
                os.remove(audiofile)
                audiofile = filename
            
            log_mel_spectrogram = self.convert2spectogram.transform(audiofile).tolist()
        except:
            print(f"Error dalam mengambil audio, kemungkinan audio ini tidak memiliki aktivitas suara")
        
        return torch.tensor(add_padding(log_mel_spectrogram, n_mels = N_MELS, max_padding = 728)).to(device)
```

# 05/11/2023
Training on 85 epochs, the result is insignificant:
```
Ground truth 0 =>  <setelah masuk kamar saya menutup pintu>
Generation 0   =>  None
Ada kesalahan saat menghitung WER
-------------------------------------------------------------------------
Ground truth 1 =>  <daging cincang yang digunakan dalam pai daging cincang tidak mengandung daging sama sekali>
Generation 1   =>  <bagian ini sedang menggugugi kekekakanananan n n n namananan nananan n n n n g g g g g g g g g g g g g g g sasasasasasa>
-------------------------------------------------------------------------
Ground truth 2 =>  <tunggu saya tunggu sebentar>
Generation 2   =>  <kudang siang mudah cuka>
-------------------------------------------------------------------------
Ground truth 3 =>  <datanglah ke rumah kapanpun kamu suka>
Generation 3   =>  <terdapat mana kamu tangan kamu ke lisakan>
-------------------------------------------------------------------------
Ground truth 4 =>  <itulah jawaban yang benar>
Generation 4   =>  <tidak bulan semarah>
-------------------------------------------------------------------------
Ground truth 5 =>  <namun ia tidak muncul untuk sesi rekaman apa pun>
Generation 5   =>  <namun tidak tidak listri melihat sebuah hari itut>
-------------------------------------------------------------------------
Ground truth 6 =>  <aku tidak akan pernah meninggalkanmu>
Generation 6   =>  <aku terkenal sediring mereka aku dalam belam>
-------------------------------------------------------------------------
Ground truth 7 =>  <sesekali ikutlah acara kami>
Generation 7   =>  <sekelah ini juga terkata pangun>
-------------------------------------------------------------------------
Ground truth 8 =>  <akan kulakukan yang kubisa>
Generation 8   =>  <apakah kamu terlu benarikan>
-------------------------------------------------------------------------
Ground truth 9 =>  <tadi malam saya tidur awal karena lelah>
Generation 9   =>  <perja hari suka berumah dari orang anda>
-------------------------------------------------------------------------
Ground truth 10 =>  <tolong beritahu saya nama dan nomor penerbangan anda>
Generation 10   =>  <lampun kemarin saya sangat mengenarkan dengan negarananananan>
-------------------------------------------------------------------------
Ground truth 11 =>  <baru kemarin saya tahu hal tersebut>
Generation 11   =>  <bagaimana orang saya untuk mary saya turus>
-------------------------------------------------------------------------
Ground truth 12 =>  <saya meminjam telepon dari pak kimura>
Generation 12   =>  <saya memberikan saya menyembaik di kuar>
-------------------------------------------------------------------------
Ground truth 13 =>  <rumah adalah tempat kita pulang>
Generation 13   =>  <penghaman adalah yang tom ke toko atah>
-------------------------------------------------------------------------
Ground truth 14 =>  <meskipun dia telah tiada aku tetap mencintainya lebih dari apapun juga>
Generation 14   =>  <telah saya tidak membuat setuah memperinya kamu tidak setapa dia mengakukan tidak buku tidak mudah>
-------------------------------------------------------------------------
Ground truth 15 =>  <karena teleponnya tidak pernah secara fisik ditutup sambungannya tetap bebas pulsa>
Generation 15   =>  <karena itu dia menyeratikan sebagai pertikan mesusik>
-------------------------------------------------------------------------
```
