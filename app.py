import os
import io
import base64
import re
import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --- PyTorch Model Definition (MUST match the one in train_and_save_model.py) ---
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Configuration and Initialization ---
app = Flask(__name__)
MODEL_PATH = 'mnist_cnn.pth'
device = torch.device("cpu") # Use CPU for deployment simplicity

# Define the exact transformation used in training, including the inversion.
# Lambda function: 1 - x inverts the tensor from (0=white, 1=black) to (1=white, 0=black)
# which is the correct MNIST format after normalization.
transform = transforms.Compose([
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)), 
    # CRITICAL: Invert tensor colors (white digit on black background)
    transforms.Lambda(lambda x: 1.0 - x)
])

# Load the model state dictionary
try:
    model = DigitRecognizer().to(device)
    # Load weights saved from the training script
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print("PyTorch model loaded successfully.")
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    model = None

# --- API Routes ---

@app.route('/')
def index():
    """Serves the main frontend page."""
    # Assuming index.html is in the 'templates' folder
    return render_template('index.html') 

def preprocess_image(data_url):
    """
    Decodes the base64 image data from the canvas and preprocesses it 
    into a PyTorch tensor, applying the necessary inversion.
    """
    # Remove the data URL header ("data:image/png;base64,")
    img_b64 = re.search(r'base64,(.*)', data_url).group(1)
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(img_b64)
    
    # Load image using PIL and convert to grayscale
    img = Image.open(io.BytesIO(img_bytes)).convert("L") 

    # 1. Resize/Resample (Canvas is 280x280, model expects 28x28)
    # Note: Resizing is handled by the transform pipeline, but this ensures PIL is ready.
    
    # 2. PyTorch Transform (applies Resize, ToTensor, Normalize, and Inversion)
    tensor_img = transform(img)
    
    # 3. Add batch dimension (1, 1, 28, 28)
    return tensor_img.unsqueeze(0)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if model is None:
        return jsonify({'error': 'Model not available.'}), 500

    try:
        data = request.get_json(force=True)
        img_data_url = data['image_data']
        
        # Preprocess the image
        input_tensor = preprocess_image(img_data_url).to(device)
        
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            
        # Apply Softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)[0]
        
        # Get the predicted class (digit)
        predicted_digit = torch.argmax(probabilities).item()
        
        response = {
            'prediction': predicted_digit,
            'confidence': f"{probabilities[predicted_digit].item():.4f}",
            'probabilities': {i: f"{p.item():.4f}" for i, p in enumerate(probabilities)}
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # You must run the train_and_save_model.py script first to generate mnist_cnn.pth
    app.run(debug=True, host='0.0.0.0', port=5000)