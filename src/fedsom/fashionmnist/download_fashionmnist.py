import torch
from torchvision import datasets
import pandas as pd
from pathlib import Path    
import torchvision 


data_dir = Path('../../../sandbox/data/fashionmnist/')
if not data_dir.exists():
    data_dir.mkdir(parents=True)

# Download Fashion MNIST dataset
trainset = datasets.FashionMNIST(root=data_dir, train=True, download=True)#, transform=transform)
testset = datasets.FashionMNIST(root=data_dir, train=False, download=True)#, transform=transform)

# Define a function to convert data to CSV
def convert_to_csv(dataset, csv_filename):
    data = dataset.data.numpy()/255.
    labels = dataset.targets.numpy()
    flattened_data = data.reshape(data.shape[0], -1)
   
    df = pd.DataFrame(flattened_data)
    df['label'] = labels
   
    df.to_csv(csv_filename, index=False)

# Convert train and test sets to CSV
convert_to_csv(trainset, data_dir / 'fashion_mnist_train.csv')
convert_to_csv(testset, data_dir / 'fashion_mnist_test.csv')