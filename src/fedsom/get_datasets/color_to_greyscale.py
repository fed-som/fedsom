import torch
import torch.nn.functional as F

def rgb_to_grayscale(image_tensor):
    # Ensure the input tensor has shape (C, H, W) where C is the number of channels
    if len(image_tensor.shape) != 3:
        raise ValueError("Input tensor must have shape (C, H, W)")

    # Convert RGB to grayscale using the luminance formula
    grayscale_tensor = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]

    return grayscale_tensor

# Example usage:
# Assuming img_tensor is a 28x28 color image tensor with shape (3, 28, 28)
img_tensor = torch.rand(3, 28, 28)  # Replace this with your actual image tensor

# Convert to grayscale
grayscale_img = rgb_to_grayscale(img_tensor)

# Display the shapes before and after conversion
print("Original shape:", img_tensor.shape)
print("Grayscale shape:", grayscale_img.shape)