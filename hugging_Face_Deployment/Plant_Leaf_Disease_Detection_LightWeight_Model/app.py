import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models
from PIL import Image
import gradio as gr
from collections import Counter
from timm import create_model
import os
import requests

# Define the GhostNetV2 model class
class GhostNetV2(nn.Module):
    def __init__(self, num_classes):
        super(GhostNetV2, self).__init__()
        self.model = create_model('ghostnetv2_100', pretrained=True)
        # Replace the final layer
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define the ResNet18 model class
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = torchvision_models.resnet18(pretrained=True)
        # Freeze the layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Replace the final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define the MobileViT model class
class MobileViT(nn.Module):
    def __init__(self, num_classes):
        super(MobileViT, self).__init__()
        self.model = create_model('mobilevit_s', pretrained=True)
        num_features = self.model.head.in_features
        # Replace the final layer
        self.model.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.model.head(x)
        return x

def load_model(checkpoint_path, model_type):
    print(f"Loading model from {checkpoint_path} of type {model_type}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(f"Checkpoint loaded: {checkpoint.keys()}")

    num_classes = checkpoint['num_classes']
    
    if model_type == 'ghostnetv2':
        model = GhostNetV2(num_classes=num_classes)
    elif model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif model_type == 'mobilevit':
        model = MobileViT(num_classes=num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model {model_type} loaded successfully")
    return model, checkpoint['class_to_idx']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model_paths = {
    "ghostnetv2": "model_pth/PlantVillage_CNN_checkpoint_GhostNetV2_V1.pth",
    "resnet18": "model_pth/PlantVillage_CNN_checkpoint_Resnet18_V1.pth",
    "mobilevit": "model_pth/PlantVillage_CNN_checkpoint_MobileVit_V1.pth"
}

loaded_models = []
class_to_idx = None

for model_type, path in model_paths.items():
    try:
        model, class_to_idx = load_model(path, model_type)
        loaded_models.append(model)
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")

if class_to_idx is None:
    raise ValueError("Class to index mapping is not loaded properly.")
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"Class to index mapping: {idx_to_class}")

test_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def check_internet_connection(timeout=5):
    try:
        requests.get("http://www.google.com", timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

def predict_leaf_disease(image, model_choice):
    try:
        print("Starting prediction")
        img = Image.fromarray(image).convert("RGB")
        img = test_transform(img).unsqueeze(0).to(device)
        print("Image transformed")

        model = loaded_models[model_choice]
        model_name = ["GhostNetV2", "ResNet18", "MobileViT"][model_choice]

        print(f"Running model: {model_name}")
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, predicted = probabilities.topk(1, dim=1)
        class_idx = predicted.item()
        class_name = idx_to_class[class_idx]
        confidence = top_prob.item() * 100

        result = f"Predicted Disease: {class_name}"
        confidence_result = f"{model_name} Confidence: {confidence:.2f}%"

        print("Prediction complete")
        return result, confidence_result, ""
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}", "", ""

# Custom CSS for styling
css = """
body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
}

h1 {
    color: #333333;
    text-align: center;
    margin-top: 20px;
}

.gradio-container {
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    padding: 20px;
}

.gradio-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
}

.gradio-button:hover {
    background-color: #45a049;
}

.gradio-checkmark {
    margin-top: 10px;
}

.gradio-textbox, .gradio-markdown {
    margin-top: 10px;
}

.gradio-textbox textarea, .gradio-markdown div {
    font-size: 16px; /* Increase the font size */
    font-weight: bold; /* Make the font bold */
    color: #333333; /* Font color */
}
"""

# Create Gradio interface
interface = gr.Interface(
    fn=predict_leaf_disease,
    inputs=[
        gr.Image(type="numpy"),
        gr.Radio(label="Select Model", choices=["GhostNetV2", "ResNet18", "MobileViT"], type="index")
    ],
    outputs=[
        gr.Textbox(label="Predicted Disease", lines=2),
        gr.Textbox(label="Confidence Level", lines=3)
    ],
    title="Plant Disease Detection",
    description="Upload a leaf image and select a model to predict its disease and get the confidence level of the prediction.",
    css=css
)

interface.launch(share=True)
