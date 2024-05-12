import os
import librosa
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

data_dir = "C:\\Users\\hp\\Desktop\\project\\dataset\\TESS Toronto emotional speech set data"
features, labels = [], []

for emotion in tqdm(os.listdir(data_dir)):
    for file in os.listdir(os.path.join(data_dir, emotion)):
        file_path = os.path.join(data_dir, emotion, file)

        # Add a try-except block to catch file loading errors
        try:
            # Load audio file with a specific sample rate (e.g., 22050)
            waveform, sample_rate = librosa.load(file_path, sr=22050)
            features.append(waveform)
            labels.append(emotion[4:])
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

# Rest of your code remains the same...


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
train_files, test_files, train_labels, test_labels = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# ... rest of your code ...

# Custom dataset class
# Custom dataset class
class SpeechEmotionDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        waveform, sample_rate = librosa.load(file_path, sr=None)

        # Convert NumPy array to PyTorch tensor
        waveform = torch.tensor(waveform)

        # Apply padding to make all waveforms have the same length
        max_len = max(len(waveform), 80000)
        waveform = torch.nn.functional.pad(waveform, (0, max_len - len(waveform)))

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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define a simple model (replace this with your actual model architecture)
class EmotionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(EmotionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Calculate the flattened size dynamically based on a dummy input
        self.dummy_input = torch.randn(1, 1, 128, 73)  # Adjust the size based on your input spectrogram size
        self.calculate_flattened_size()
        self.fc1 = torch.nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def calculate_flattened_size(self):
        with torch.no_grad():
            x = self.pool(torch.relu(self.conv1(self.dummy_input)))
            self.flattened_size = x.view(1, -1).size(1)

# Instantiate the model and define loss function and optimizer
model = EmotionModel(num_classes=len(label_encoder.classes_))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
# print(train_loader)
num_epochs = 25
all_predictions, all_labels = [], []
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(inputs.unsqueeze(1))  # Add a channel dimension for the Mel spectrogram
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')
print(f'Final Loss: {loss.item():6f}')
# Calculate accuracy using scikit-learn
accuracy = accuracy_score(all_labels, all_predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')            