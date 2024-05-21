import torch
from torchvision import datasets, transforms
import pandas as pd
from pathlib import Path  


path = Path('./kuzushiji_data/')
if not path.exists():
    path.mkdir(parents=True)


# Set a seed for reproducibility
torch.manual_seed(42)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download the Kuzushiji-MNIST dataset
train_dataset = datasets.KMNIST(root='./kuzushiji_train_data', train=True, download=True, transform=transform)
data = train_dataset.data.numpy()
labels = train_dataset.targets.numpy()
data = data / 255.0
data = data.reshape(data.shape[0], -1)
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(path / 'kuzushiji_mnist_normalized_train.csv', index=False)


test_dataset = datasets.KMNIST(root='./kuzushiji_test_data', train=False, download=True, transform=transform)
data = test_dataset.data.numpy()
labels = test_dataset.targets.numpy()
data = data / 255.0
data = data.reshape(data.shape[0], -1)
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(path / 'kuzushiji_mnist_normalized_test.csv', index=False)


print("Normalized Kuzushiji-MNIST data saved as kuzushiji_mnist_normalized.csv")