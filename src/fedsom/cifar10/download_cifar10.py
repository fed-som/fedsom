import torchvision
import torchvision.transforms as transforms
from pathlib import Path 
import numpy as np
import pandas as pd
from torchvision import datasets



# # Define a transform to preprocess the data (optional)
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Converts images to PyTorch tensors
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
# ])


data_dir = Path('../../../sandbox/data/cifar10')
trainset = datasets.CIFAR10(root=data_dir, train=True, download=True)#, transform=transform)
testset = datasets.CIFAR10(root=data_dir, train=False, download=True)#, transform=transform)

# Define a function to flatten and convert images to CSV
def flatten_and_save_as_csv(dataset, csv_filename):
    images = dataset.data / 255.0
    labels = dataset.targets
   
    num_images = len(images)
    flattened_images = images.reshape(num_images, -1)
   
    df = pd.DataFrame(flattened_images)
    df['label'] = labels
   
    df.to_csv(csv_filename, index=False)




# Convert train and test sets to CSV
flatten_and_save_as_csv(trainset, data_dir / 'cifar10_train.csv')
flatten_and_save_as_csv(testset, data_dir / 'cifar10_test.csv')



# Download CIFAR-10 dataset
# trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,download=True, transform=transform)











