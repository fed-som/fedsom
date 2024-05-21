import os
import pandas as pd
from urllib.request import urlretrieve
import numpy as np

# Function to download and load data for a specific class
def download_quickdraw_data(class_name, num_samples_per_class=10000, output_dir='quickdraw_data'):
    os.makedirs(output_dir, exist_ok=True)
   
    # URL for the Quick, Draw! dataset
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
   
    # Download the data file for the specified class
    url = base_url + f"{class_name}.npy"
    file_path = os.path.join(output_dir, f"{class_name}.npy")
    urlretrieve(url, file_path)
   
    # Load the data into a NumPy array
    data = np.load(file_path)
   
    # Take a random subset of samples from the class
    if len(data) > num_samples_per_class:
        indices = np.random.choice(len(data), num_samples_per_class, replace=False)
        data = data[indices]
   
    # Create a DataFrame with features and labels
    df = pd.DataFrame(data, columns=[f"pixel_{i}" for i in range(data.shape[1])])
    df['label'] = class_name
   
    return df

# List of classes you want to download
classes_to_download = ['apple', 'banana', 'cat', 'dog']

# Number of samples per class (adjust as needed)
num_samples_per_class = 10000

# Output directory for saving the CSV files
output_directory = 'quickdraw_data_csv'

# Download and save data for each class
for class_name in classes_to_download:
    class_df = download_quickdraw_data(class_name, num_samples_per_class, output_directory)
    class_df.to_csv(os.path.join(output_directory, f"{class_name}_dataset.csv"), index=False)

print("Quick, Draw! datasets saved as CSV files.")