import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import random_split # Import for the dynamic split

# --- 1. Model Definition (CNN Architecture) ---
class DigitRecognizer(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for digit recognition.
    Designed for 28x28 grayscale images (like MNIST).
    """
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer: 64 channels * 7*7 (after 2 max pools: 28/2/2 = 7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 output classes (0-9)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the image tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output logits (no softmax here, as CrossEntropyLoss handles it)
        return x

# --- 2. Data Loading and Preprocessing (TRAIN FOLDER ONLY) ---
def load_data(use_custom=False):
    """
    Loads data either from the standard MNIST set or exclusively from the 
    custom_dataset/train folder, performing an automatic split.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if use_custom:
        print("Loading ALL custom data from './custom_dataset/train' and performing an 80/20 split...")
        CUSTOM_DATA_ROOT = './custom_dataset'

        # Load the entire dataset from the 'train' folder
        train_data_all = datasets.ImageFolder(
            root=os.path.join(CUSTOM_DATA_ROOT, 'train'),
            transform=transform
        )
        
        if len(train_data_all) == 0:
            raise RuntimeError("No custom data found in './custom_dataset/train'. Did you forget to run the segmentation script and label the files?")

        # --- Automatic 80% / 20% Split ---
        total_size = len(train_data_all)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size
        
        # Perform the random split on the single dataset object
        train_data, test_data = random_split(
            train_data_all, [train_size, test_size],
            generator=torch.Generator().manual_seed(42) # Ensure repeatable split
        )
        print(f"Dataset Split: Training on {len(train_data)} samples, Validating on {len(test_data)} samples.")
        
    else:
        print("Loading standard PyTorch MNIST dataset...")
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# --- 3. Training Function (UNMODIFIED) ---
def train_model(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# --- 4. Main Execution (UNMODIFIED) ---
if __name__ == '__main__':
    # Determine device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS) for acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # --- CHANGE THIS TO TRUE TO USE YOUR CUSTOM DATA ---
    USE_CUSTOM_DATA = True 
    # --------------------------------------------------

    # Now load_data uses the ImageFolder logic to find your labeled data
    train_loader, _ = load_data(use_custom=USE_CUSTOM_DATA) 

    # Instantiate the model and move to device
    model = DigitRecognizer().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model (adjust epochs as needed)
    print("Starting training...")
    # We are using 10 epochs for your custom, small dataset.
    train_model(model, device, train_loader, optimizer, criterion, epochs=10) 
    
    # Save the model state dictionary
    MODEL_FILENAME = 'mnist_cnn.pth'