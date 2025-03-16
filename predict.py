import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import json
from torchvision import models
import os
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained deep learning model")
    
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--name_file', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath, weights_only=False)
    model = models.mobilenet_v2(pretrained=True)
    model.name ="mobilenet_v2"

    for param in model.parameters():
        param.requires_grad =False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB") 
    
    image = transform(image)
    
    numpy_image = np.array(image)
     
    return torch.unsqueeze(image, 0)
def predict(image, model, cat_to_name=None, topk=5, device="cpu"):
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    probab = torch.exp(output)
    probs, indices = probab.topk(topk, dim=1)
    
    list_probs = probs.tolist()[0]
    list_indices = indices.tolist()[0]
    
    index_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [index_map[idx] for idx in list_indices]

    # Convert class indices to actual names if `cat_to_name` is provided
    if cat_to_name:
        labels = [cat_to_name[c] for c in classes]
        return list_probs, labels
    
    return list_probs, classes
    
def print_predictions(probabilities, classes, cat_to_name=None):
    labels = [cat_to_name.get(c, f"Class {c}") for c in classes] if cat_to_name else classes

    for i, (probab, label, c) in enumerate(zip(probabilities, labels, classes), 1):
        print(f"{i}) {probab*100:.2f}% - {label.title()} (Class {c})")

def main():
    args = get_input_args()

    with open(args.name_file, 'r') as f:
                cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    image = process_image(args.image_path)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model.to(device)

    probs, classes = predict(image, model, cat_to_name, args.top_k, device)
    print_predictions(probs, classes)
    
      
if __name__ == '__main__':
    main()    