import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models
import torch.nn as nn

# Define constants
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
    transforms.ToTensor(),
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize a pretrained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, NUM_CLASSES)
)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(NUM_EPOCHS):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# Save the trained model (optional)
torch.save(model.state_dict(), 'custom_resnet_cifar10_model.pth')










# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# from torchvision import models

# # Define constants
# NUM_CLASSES = 10
# BATCH_SIZE = 64
# NUM_EPOCHS = 10

# # Load CIFAR-10 dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
#     transforms.ToTensor(),
# ])

# train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Initialize a pretrained ResNet model
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

# # Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# for epoch in range(NUM_EPOCHS):
#     for images, labels in train_loader:
#         print(images.shape)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# # Save the trained model (optional)
# torch.save(model.state_dict(), 'resnet_cifar10_model.pth')