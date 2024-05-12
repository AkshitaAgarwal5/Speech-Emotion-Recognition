import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, Resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

def calculate_flattened_size(self):
    with torch.no_grad():
        x = self.pool(torch.relu(self.conv1(self.dummy_input)))
        self.flattened_size = x.view(x.size(0), -1).size(1)

def collate_fn(batch):
    waveforms, labels = zip(*batch)

    # Find the maximum length in the batch
    max_len = max(w.size(1) for w in waveforms)

    # Pad or truncate each waveform to have the same length
    padded_waveforms = [
        F.pad(w, (0, max_len - w.size(1)), value=0)
        if w.size(1) < max_len
        else w[:, :max_len]
        for w in waveforms
    ]

    return torch.stack(padded_waveforms), torch.tensor(labels, dtype=torch.long)




data_dir = "C:\\Users\\hp\\Desktop\\project\\dataset\\TESS Toronto emotional speech set data"
features, labels = [], []
for emotion in tqdm(os.listdir(data_dir)):
    for file in os.listdir(os.path.join(data_dir, emotion)):
        file_path = os.path.join(data_dir, emotion, file)
        features.append(file_path)
        labels.append(emotion[4:])

# Encode labels and Split the dataset into training and testing sets
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
train_files, test_files, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Custom dataset class
class SpeechEmotionDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            waveform, sample_rate = torchaudio.load(self.files[idx])
        except Exception as e:
            print(f"Error loading file {self.files[idx]}: {e}")
            # Return a default tensor if loading fails
            return torch.zeros(1, 80000), torch.tensor(0, dtype=torch.long)

        # Handle cases where waveform size is not 80000
        if waveform.size(1) != 80000:
            print(f"File with incorrect size: {self.files[idx]}")
            # Apply padding or truncation to make all waveforms have the same length
            waveform = torch.nn.functional.pad(waveform, (0, 80000 - waveform.size(1)))
            # Alternatively, you can truncate the waveform
            # waveform = waveform[:, :80000]

        if self.transform:
            waveform = self.transform(waveform)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, label

transform = torch.nn.Sequential(
    Resample(orig_freq=44100, new_freq=8000),
    MelSpectrogram(n_fft=800, hop_length=200, n_mels=128, normalized=True),
)

# Create datasets and dataloaders
train_dataset = SpeechEmotionDataset(train_files, train_labels, transform=transform)
test_dataset = SpeechEmotionDataset(test_files, test_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)



class EmotionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(EmotionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dummy_input = torch.randn(1, 1, 128, 73)  # Adjust the size based on your input spectrogram size
        self.calculate_flattened_size()
        self.fc1 = torch.nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def calculate_flattened_size(self):
        with torch.no_grad():
            x = self.pool(torch.relu(self.conv1(self.dummy_input)))
            self.flattened_size = x.view(x.size(0), -1).size(1)


# Instantiate the model and define loss function and optimizer
model = EmotionModel(num_classes=len(label_encoder.classes_))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
num_epochs = 25
all_predictions, all_labels = [], []
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        if inputs is None:
            continue
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    scheduler.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}')

# Calculate accuracy using scikit-learn
accuracy = accuracy_score(all_labels, all_predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model on the test set
model.eval()
# Save the trained model if needed
torch.save(model.state_dict(), 'emotion_model.pth')

def plot_frequency_graph(audio_path, emotion):
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(numpy.abs(librosa.stft(y)), ref=numpy.max)
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(emotion.upper())
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
x=100
L = ["angry", "disgust", "Fear", "happy", "neutral", "Pleasant_suprise", "sad"]
for emotion in L:
    plot_frequency_graph(features[x], emotion)
    x+=200
    plot_frequency_graph(features[x], emotion)
    x+=200

# Function to record audio
def record_audio(duration=5, sample_rate=8000, filename='user_input.wav'):
    print("Recording...")
    audio_data = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=numpy.int16)
    sounddevice.wait()
    wavfile.write(filename, sample_rate, audio_data.squeeze())
    print(f"Recording saved as {filename}")

# Function to predict emotion from user input
def predict_emotion(model, input_file, transform):
    waveform, sample_rate = torchaudio.load(input_file)
    # Apply padding to make the waveform length compatible with the model
    max_len = max(waveform.size(1), 80000)
    waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.size(1)))
    # Apply the same transformation used during training
    with torch.no_grad():
        input_data = transform(waveform)
        input_data = input_data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        output = model(input_data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        emotion_idx = torch.argmax(probabilities, dim=1).item()
    return label_encoder.classes_[emotion_idx]

record_audio(filename='user_input.wav') # Record audio from the user
user_input_emotion = predict_emotion(model, 'user_input.wav', transform) # Use the trained model to predict emotion
print(f"Predicted Emotion: {user_input_emotion}") # Display the predicted emotion

audio_file_path = r"C:\\Users\\hp\\Desktop\\project\\dataset\\TESS Toronto emotional speech set data\\input.wav"
plot_frequency_graph(audio_file_path, user_input_emotion)    