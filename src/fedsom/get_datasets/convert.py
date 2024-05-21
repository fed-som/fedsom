

# import os
# import pandas as pd
# from PIL import Image
# import numpy as np

# # Directory where the Chars74K dataset is expanded
# dataset_dir = "./"

# # Output CSV file path
# output_csv_path = "not_mnist.csv"

# # List to store pixel values and corresponding labels
# data = []

# # Function to convert an image to a coarser numerical representation
# def image_to_coarse_vector(image_path, target_size=(40, 40)):
#     image = Image.open(image_path).convert("L")  # Convert to grayscale
#     resized_image = image.resize(target_size, Image.ANTIALIAS)
#     vector = np.array(resized_image).flatten() / 255.0  # Normalize by dividing by 255
#     return vector

# # Iterate through subdirectories (each subdirectory corresponds to a class)
# for class_name in os.listdir(dataset_dir):
#     class_dir = os.path.join(dataset_dir, class_name)
#     print(class_dir)
#     if os.path.isdir(class_dir):
#         for filename in os.listdir(class_dir):
#             if filename.endswith(".png"):
#                 image_path = os.path.join(class_dir, filename)
#                 vector = image_to_coarse_vector(image_path)
#                 data.append(tuple(vector) + (class_name,))

# # Determine the number of pixels based on the first image
# num_pixels = len(data[0]) - 1
# print(num_pixels)

# # Create column names for pixels
# pixel_columns = [f"pixel_{i}" for i in range(num_pixels)]

# # Convert the data to a DataFrame
# print('Creating dataframe...')
# df = pd.DataFrame(data, columns=pixel_columns + ["label"])

# # Save the DataFrame to a CSV file
# df.to_csv(output_csv_path, index=False)

# print(f"Coarse normalized numerical vectors saved as {output_csv_path}")





















import os
import pandas as pd
from PIL import Image
import numpy as np

# Directory where the Chars74K dataset is expanded
dataset_dir = "./"

# Output CSV file path
output_csv_path = "not_mnist.csv"

# List to store pixel values and corresponding labels
data = []

# Function to convert an image to numerical vectors
def image_to_vector(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    vector = np.array(image).flatten() / 255.0  # Normalize by dividing by 255
    return vector

# Iterate through subdirectories (each subdirectory corresponds to a class)
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    print(class_dir)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(class_dir, filename)
                vector = image_to_vector(image_path)
                data.append(tuple(vector) + (class_name,))

# Determine the number of pixels based on the first image
num_pixels = len(data[0]) - 1
print(num_pixels)
# Create column names for pixels
pixel_columns = [f"pixel_{i}" for i in range(num_pixels)]

# Convert the data to a DataFrame
print('Converting data to dataframe...')
df = pd.DataFrame(data, columns=pixel_columns + ["label"])

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print(f"Normalized numerical vectors saved as {output_csv_path}")












# import os
# import pandas as pd
# from PIL import Image
# import numpy as np

# # Directory where the Chars74K dataset is expanded
# dataset_dir = "./"

# # Output CSV file path
# output_csv_path = "chars74k_numerical_vectors.csv"

# # List to store image paths and corresponding labels
# data = []

# # Function to convert an image to a numerical vector
# def image_to_vector(image_path):
#     image = Image.open(image_path).convert("L")  # Convert to grayscale
#     vector = np.array(image).flatten()
#     return vector

# # Iterate through subdirectories (each subdirectory corresponds to a class)
# for class_name in os.listdir(dataset_dir):
#     print(class_name)
#     class_dir = os.path.join(dataset_dir, class_name)
#     if os.path.isdir(class_dir):
#         for filename in os.listdir(class_dir):
#             if filename.endswith(".png"):
#                 image_path = os.path.join(class_dir, filename)
#                 data.append((image_path, class_name))

# # Convert the data to a DataFrame
# df = pd.DataFrame(data, columns=["image_path", "label"])

# # Apply the image_to_vector function to each image path
# df["vector"] = df["image_path"].apply(image_to_vector)

# # Save the DataFrame to a CSV file
# df.to_csv(output_csv_path, index=False)

# print(f"Numerical vectors saved as {output_csv_path}")