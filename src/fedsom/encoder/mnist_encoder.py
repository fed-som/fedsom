# Create a random tensor of shape (64, 1, 28, 28)
tensor_original = torch.randn(64, 1, 28, 28)

# Flatten the tensor to shape (64, 784)
tensor_flat = tensor_original.view(64, -1)

# Reshape it back to (64, 1, 28, 28)
tensor_restored = tensor_flat.view(64, 1, 28, 28)

# Print shapes to verify
print("Original shape:", tensor_original.shape)
print("Flattened shape:", tensor_flat.shape)
print("Restored shape:", tensor_restored.shape)

# Check if the tensors are equal
print("Are tensors equal?", torch.all(tensor_original == tensor_restored))




import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize LeNet-5 model
net = LeNet5()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the model (you may need to run this for several epochs for good performance)
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

# Save the trained model
torch.save(net.state_dict(), 'lenet5_mnist.pth')