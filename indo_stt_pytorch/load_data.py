dataset_url = ""
dataset_filename = os.path.join(os.getcwd(), "data.tar.bz2")
save_dir = "./datasets/"

response = requests.get(dataset_url)
with open(dataset_filename, 'wb') as f:
    f.write(response.content)

with tarfile.open(dataset_filename, 'r:bz2') as archive:
    archive.extractall(save_dir)

id_to_text = {}
with open(os.path.join(save_dir, "metadata.csv"), encoding="utf-8") as f:
    csv_reader = csv.reader(f)
    for line in csv_reader:
        id = line[0]
        text = line[2]
        id_to_text[id] = text

class CustomDataset(Dataset):
    def __init__(self, wavs, id_to_text, maxlen=50):
        self.data = []
        for w in wavs:
            id = w.split("/")[-1].split(".")[0]
            if len(id_to_text[id]) < maxlen:
                self.data.append({"audio": w, "text": id_to_text[id]})
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

wavs = glob("{}/**/*.wav".format(save_dir), recursive=True)

dataset = CustomDataset(wavs, id_to_text)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", "," "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

max_target_len = 200 
data = get_data(wavs, id_to_text, max_target_len)
vectorizer = VectorizeChar(max_target_len)
print("vocab size", len(vectorizer.get_vocabulary()))

class CustomDataset(Dataset):
    def __init__(self, data, vectorizer):
        self.data = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]["audio"]
        text = self.data[idx]["text"]
        audio = path_to_audio(audio_path)
        text_vector = self.vectorizer(text)
        return {"source": audio, "target": text_vector}

def path_to_audio(path):
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
    return spectrogram

train_split = int(len(data) * 0.99)
train_data = data[:train_split]
test_data = data[train_split:]

train_dataset = CustomDataset(train_data, vectorizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = CustomDataset(test_data, vectorizer)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
