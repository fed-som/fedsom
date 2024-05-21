import os
import pandas as pd
from urllib.request import urlretrieve
from zipfile import ZipFile

# URL to the Chars74K dataset ZIP file
dataset_url = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"

# Directory to store the downloaded and extracted data
download_dir = "chars74k_data"

# Ensure the directory exists or create it
os.makedirs(download_dir, exist_ok=True)

# Download and extract the dataset
download_path = os.path.join(download_dir, "EnglishFnt.tgz")
urlretrieve(dataset_url, download_path)

# Extract the contents of the ZIP file
with ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

# Define a function to read the images and labels and convert them to a DataFrame
def load_chars74k_data(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                label = os.path.basename(root)
                image_path = os.path.join(root, file)
                data.append((image_path, label))
    return pd.DataFrame(data, columns=["image_path", "label"])

# Load the data into a DataFrame
df = load_chars74k_data(download_dir)

# Save the DataFrame to a CSV file
df.to_csv('chars74k_dataset.csv', index=False)

print("Chars74K dataset saved as chars74k_dataset.csv")
