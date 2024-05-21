import pandas as pd
import requests
from io import StringIO

# URL of the Sign Language MNIST dataset CSV file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/american-sign-language-mnist/sign_mnist_train.csv"

# Download the CSV file
response = requests.get(url)
data_csv = StringIO(response.text)

# Create a DataFrame
df = pd.read_csv(data_csv)

# Save the DataFrame to a CSV file
df.to_csv('slmnist_dataset.csv', index=False)

print("SLMNIST dataset saved as slmnist_dataset.csv")