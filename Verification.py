import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set this before importing pyplot
import matplotlib.pyplot as plt

# Load your tensor
X = torch.load('X.pt')  # Replace with your tensor file path


def visualize_image(tensor, index, save_path='reconstructed_images'):
    # Create directory if it doesn't exist
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 0 <= index < len(tensor):
        # Reshape and convert to numpy for visualization
        img = tensor[index].reshape(28, 28).numpy()

        # Create figure
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Image at index {index}')

        # Save and close
        file_path = f'{save_path}/image_{index}.png'
        plt.savefig(file_path)
        plt.close()

        print(f"Image saved at: {file_path}")
    else:
        print(f"Error: Index {index} is out of range. Valid range is 0-{len(tensor) - 1}")


# Example usage:
index = 0  # Change this to see different images
visualize_image(X, index)

# To see multiple specific indices
indices = [0, 10, 20, 30, 40]  # Change these to the indices you want to see
for idx in indices:
    visualize_image(X, idx)

print(torch.load("Y.pt")[40])