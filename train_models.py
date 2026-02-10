# your_project_root/train_models.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import copy

# --- 1. Configuration ---
DATA_DIR = 'dataset'
MODEL_SAVE_DIR = 'models'
BATCH_SIZE = 32
NUM_EPOCHS = 15 # You might need more epochs depending on dataset size
LEARNING_RATE = 0.001
NUM_CLASSES = 5 # dolphin, fish, lobster, octopus, sea_horse
CLASS_NAMES = ['dolphin', 'fish', 'lobster', 'octopus', 'sea_horse'] # Ensure order matches ImageFolder
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# --- 2. Data Preprocessing ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Loading data...")
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4 if DEVICE.type == 'cuda' else 0) # num_workers can be adjusted
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# Verify class names order
print(f"Detected class names: {image_datasets['train'].classes}")
# Ensure your CLASS_NAMES list above matches this order
if not all(a == b for a, b in zip(CLASS_NAMES, image_datasets['train'].classes)):
    print("WARNING: Class names order mismatch. Please verify CLASS_NAMES in script.")
    CLASS_NAMES = image_datasets['train'].classes # Use the order detected by ImageFolder

# --- 3. Training Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- 4. Model Initialization and Training ---

# Model 1: MobileNetV2
print("\n--- Training MobileNetV2 ---")
model_mobilenet = models.mobilenet_v2(pretrained=True)
for param in model_mobilenet.parameters():
    param.requires_grad = False # Freeze feature extractor
num_ftrs_mobilenet = model_mobilenet.classifier[1].in_features
model_mobilenet.classifier[1] = nn.Linear(num_ftrs_mobilenet, NUM_CLASSES) # Replace classifier head
model_mobilenet = model_mobilenet.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer_mobilenet = optim.Adam(model_mobilenet.classifier.parameters(), lr=LEARNING_RATE)
exp_lr_scheduler_mobilenet = optim.lr_scheduler.StepLR(optimizer_mobilenet, step_size=7, gamma=0.1)

model_mobilenet = train_model(model_mobilenet, criterion, optimizer_mobilenet, exp_lr_scheduler_mobilenet)
torch.save(model_mobilenet.state_dict(), os.path.join(MODEL_SAVE_DIR, 'mobilenetv2_marine_classifier.pth'))
print("MobileNetV2 model saved.")

# Model 2: EfficientNet-B0
print("\n--- Training EfficientNet-B0 ---")
# EfficientNet-B0 is part of torchvision models, often requiring specific weights or custom import
# For torchvision < 0.9, you might need to install 'efficientnet_pytorch' package.
# For torchvision >= 0.9, it's available as 'models.efficientnet_b0'.
try:
    model_efficientnet = models.efficientnet_b0(pretrained=True)
    # Freeze feature extractor
    for param in model_efficientnet.parameters():
        param.requires_grad = False
    # Replace classifier head
    num_ftrs_efficientnet = model_efficientnet.classifier[1].in_features
    model_efficientnet.classifier[1] = nn.Linear(num_ftrs_efficientnet, NUM_CLASSES)
    model_efficientnet = model_efficientnet.to(DEVICE)

    optimizer_efficientnet = optim.Adam(model_efficientnet.classifier.parameters(), lr=LEARNING_RATE)
    exp_lr_scheduler_efficientnet = optim.lr_scheduler.StepLR(optimizer_efficientnet, step_size=7, gamma=0.1)

    model_efficientnet = train_model(model_efficientnet, criterion, optimizer_efficientnet, exp_lr_scheduler_efficientnet)
    torch.save(model_efficientnet.state_dict(), os.path.join(MODEL_SAVE_DIR, 'efficientnetb0_marine_classifier.pth'))
    print("EfficientNet-B0 model saved.")
except AttributeError:
    print("EfficientNet-B0 not found in torchvision.models. Skipping EfficientNet training.")
    print("Consider installing 'efficientnet_pytorch' or updating torchvision if you want to use it.")
    model_efficientnet = None # Set to None if not available

# Model 3: Custom Lightweight CNN (for comparison)
# This will be simpler and demonstrate a 'from scratch' approach vs transfer learning
print("\n--- Training Custom Lightweight CNN ---")
class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 112x112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 56x56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 28x28
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model_customcnn = CustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
optimizer_customcnn = optim.Adam(model_customcnn.parameters(), lr=LEARNING_RATE)
exp_lr_scheduler_customcnn = optim.lr_scheduler.StepLR(optimizer_customcnn, step_size=7, gamma=0.1)

model_customcnn = train_model(model_customcnn, criterion, optimizer_customcnn, exp_lr_scheduler_customcnn)
torch.save(model_customcnn.state_dict(), os.path.join(MODEL_SAVE_DIR, 'customcnn_marine_classifier.pth'))
print("Custom CNN model saved.")

print("\nAll models trained and saved!")