import os
import shutil
import tarfile
import gdown

def main():
    os.makedirs("audio_folder", exist_ok=True)
    
    os.makedirs("audio_folder/train", exist_ok=True)
    os.makedirs("audio_folder/valid", exist_ok=True)
    os.makedirs("batch_data", exist_ok=True)
    os.makedirs("common_voice_id", exist_ok=True)
  
    url = 'https://drive.google.com/u/3/uc?id=1NfCGqWJOxYG0XOWDj0K4fD3mPB7RWHRR'
    output = 'cv-corpus-15.0-2023-09-08-id.tar'
    gdown.download(url, output, quiet=False)
    
    with tarfile.open(output, "r") as tar_ref:
        tar_ref.extractall('.')
    
    if os.path.exists("cv-corpus-15.0-2023-09-08/id"):
        shutil.move("cv-corpus-15.0-2023-09-08/id", "common_voice_id")
    
    shutil.rmtree("cv-corpus-15.0-2023-09-08", ignore_errors=True)
    os.remove(output)

if __name__ == '__main__':
    main()
