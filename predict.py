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
    parser.add_argument('--name_file', type=str, default=None, help='Path to JSON file mapping categories to names')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, weights_only=False)
    arch = checkpoint.get('arch', 'mobilenet_v2')  # Default to mobilenet_v2
    if arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        input_size = model.classifier[0].in_features
        model.classifier = checkpoint['classifier']
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        input_size = model.classifier[0].in_features
        model.classifier = checkpoint['classifier']
    elif arch == 'efficientnet_b0': 
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        input_size = model.classifier[1].in_features
        model.classifier = checkpoint['classifier']
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint.get('class_to_idx', {})

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def process_image(image_path):
    """Process an image to be compatible with the model."""
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    return torch.unsqueeze(image, 0)  #batch dimension

def predict(image, model, cat_to_name=None, topk=5, device="cpu"):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.eval()
    image = image.to(device)
    model.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    probs = torch.exp(output) 
    top_probs, top_indices = probs.topk(topk, dim=1)
    
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()


    if isinstance(top_indices, int):
        top_indices = [top_indices]
        top_probs = [top_probs]

    # Convert indices to actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    # Convert class indices to actual names if `cat_to_name` is provided
    if cat_to_name:
        top_labels = [cat_to_name.get(str(cls), f"Class {cls}") for cls in top_classes]
        return top_probs, top_labels
    
    return top_probs, top_classes

def print_predictions(probabilities, classes):
    """Prints the predicted probabilities and corresponding class labels."""
    print("\nPredictions:")
    for i, (probab, label) in enumerate(zip(probabilities, classes), 1):
        print(f"{i}) {probab*100:.2f}% - {label}")

def main():
    args = get_input_args()

    # Load class-to-name mapping if provided
    cat_to_name = None
    if args.name_file and os.path.exists(args.name_file):
        with open(args.name_file, 'r') as f:
            cat_to_name = json.load(f)
            
    model = load_checkpoint(args.checkpoint)
    
    # Process the input image
    image = process_image(args.image_path)
    
    # Set device (CPU/GPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    probs, classes = predict(image, model, cat_to_name, args.top_k, device)

    print_predictions(probs, classes)

if __name__ == '__main__':
    main()
