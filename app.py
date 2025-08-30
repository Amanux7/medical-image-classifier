import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# Original Binary Model Class
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Multi-Class Model Class
class MultiDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4):  # Adjust to your multi num_classes
        super(MultiDiseaseClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title('Disease Detector from Chest X-rays')

# Mode selection
mode = st.selectbox('Select Mode', ['Binary (Normal vs. Pneumonia)', 'Multi-Class (Multiple Diseases)'])

if mode == 'Binary (Normal vs. Pneumonia)':
    model = PneumoniaClassifier().to(device)
    model.load_state_dict(torch.load('binary_model.pth', map_location=device))  # Original file (renamed)
    classes = ['Normal', 'Pneumonia']
else:
    model = MultiDiseaseClassifier().to(device)
    model.load_state_dict(torch.load('multi_model.pth', map_location=device))  # New file
    classes = ['Normal', 'Pneumonia', 'COVID-19', 'TB']  # Adjust to your multi classes

model.eval()

uploaded_file = st.file_uploader('Upload an X-ray image', type=['jpg', 'png', 'jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        st.write(f'Prediction: **{classes[pred]}**')