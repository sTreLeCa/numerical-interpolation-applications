import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from PIL import Image
import random

# Load the original image
input_image_path = "32exampleimage.png"
original_image = Image.open(input_image_path).convert('RGB')
original_array = np.array(original_image)

# Function to randomly remove pixels based on a specified percentage
def remove_pixels(image_array, percent_to_remove):
    modified_image_array = image_array.copy()
    num_pixels = int(image_array.shape[0] * image_array.shape[1] * percent_to_remove / 100)
    indices = list(np.ndindex(image_array.shape[:2]))
    random_indices = random.sample(indices, num_pixels)
    for x, y in random_indices:
        modified_image_array[x, y] = [0, 0, 0]  # Set pixel to black
    return modified_image_array

# Function to restore the image using RBF interpolation
def restore_image(original_array, modified_array, function, epsilon=10):
    mask = np.any(modified_array != [0, 0, 0], axis=-1)  # Mask of known pixels
    known_coords = np.argwhere(mask)
    known_values = modified_array[mask]
    
    restored_array = np.zeros_like(original_array)
    for channel in range(3):
        rbf = Rbf(known_coords[:, 1], known_coords[:, 0], known_values[:, channel], function=function, epsilon=epsilon)
        x_coords = np.arange(original_array.shape[1])
        y_coords = np.arange(original_array.shape[0])
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        restored_values = rbf(x_grid, y_grid)
        restored_array[:, :, channel] = restored_values.clip(0, 255)

    return restored_array.astype(np.uint8)

# Percentages of pixels to remove
percentages_to_remove = [10, 25, 50, 75]

# RBF functions to use for restoration
functions = ['gaussian', 'linear', 'multiquadric']

# Process each percentage
for percent_to_remove in percentages_to_remove:
    # Remove pixels
    modified_array = remove_pixels(original_array, percent_to_remove)

    # Save the modified image
    modified_image = Image.fromarray(modified_array)
    modified_image_path = f"modified_image_{percent_to_remove}percent_removed.png"
    modified_image.save(modified_image_path)

    # Restore images using different RBF functions
    restored_images = []
    for function in functions:
        restored_array = restore_image(original_array, modified_array, function)
        restored_image = Image.fromarray(restored_array)
        restored_images.append(restored_image)

    # Save restored images
    for i, function in enumerate(functions):
        restored_image_path = f"restored_image_{function}_{percent_to_remove}percent_removed.png"
        restored_images[i].save(restored_image_path)

    # Display images
    plt.figure(figsize=(20, 5))
    plt.subplot(1, len(functions) + 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, len(functions) + 2, 2)
    plt.imshow(modified_image)
    plt.title(f"Modified Image ({percent_to_remove}% removed)")

    for i, function in enumerate(functions):
        plt.subplot(1, len(functions) + 2, i + 3)
        plt.imshow(restored_images[i])
        plt.title(f"Restored Image ({function})")

    plt.show()
