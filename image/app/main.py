from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

# Define CNN model (same as training script)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Dynamically determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 150, 150)
            out = self.conv(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('model/image_classifier.pth', map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Flask app setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files['image']
        if img_file:
            image = Image.open(img_file.stream).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                pred = output.item()
                label = "Dog" if pred > 0.5 else "Cat"
                confidence = round(pred * 100 if label == "Dog" else (1 - pred) * 100, 2)

            return render_template("index.html", label=label, confidence=confidence)

    return render_template("index.html", label=None)

# ✅ Required to start the server
if __name__ == '__main__':
    app.run(debug=True, port=5050)
