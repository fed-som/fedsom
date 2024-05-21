import torch
from torchvision import datasets, transforms
import pandas as pd
from pathlib import Path  


path = Path('./fashion_mnist/')
if not path.exists():
    path.mkdir(parents=True)

# Set a seed for reproducibility
torch.manual_seed(42)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


train_dataset = datasets.FashionMNIST(root='./fashion_data_train', train=True, download=True, transform=transform)
data = train_dataset.data.numpy()
labels = train_dataset.targets.numpy()
data = data/255.
data = data.reshape(data.shape[0], -1)
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(path / 'fashion_mnist_train.csv', index=False)


test_dataset = datasets.FashionMNIST(root='./fashion_data_test', train=False, download=True, transform=transform)
data = test_dataset.data.numpy()
labels = test_dataset.targets.numpy()
data = data/255.
data = data.reshape(data.shape[0], -1)
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(path / 'fashion_mnist_test.csv', index=False)



print("Fashion MNIST data saved as fashion_mnist.csv")