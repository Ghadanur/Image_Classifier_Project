import torch
import torchvision
import argparse
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 model on a dataset")
    parser.add_argument('data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    
    return train_loader, valid_loader, train_dataset.class_to_idx

def build_model(hidden_units):
    model = models.mobilenet_v2(pretrained=True)
    input_size = model.classifier[1].in_features
    
    for param in model.features.parameters():  # Freeze only feature extractor
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def train_model(model, train_loader, valid_loader, device, epochs, learning_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                
                probab = torch.softmax(output, dim=1)
                top_p, top_class = probab.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train Loss: {train_loss/len(train_loader):.3f}.. "
              f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation Accuracy: {accuracy/len(valid_loader)*100:.2f}%")
    
    return model, optimizer

def save_checkpoint(model, save_dir, class_to_idx):
    checkpoint = {
        'arch': 'mobilenet_v2',
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': class_to_idx
    }
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at: {save_path}")

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)
    model = build_model(args.hidden_units)
    
    model, optimizer = train_model(model, train_loader, valid_loader, device, args.epochs, args.learning_rate)
    save_checkpoint(model, args.save_dir, class_to_idx)
    
if __name__ == '__main__':
    main()