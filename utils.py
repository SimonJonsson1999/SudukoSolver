import numpy as np
import torch
from typing import Union
import matplotlib.pyplot as plt
import cv2 as cv


def display_image(image: Union[torch.Tensor, np.ndarray], window_name: str = 'image') -> None:
    """
    Display an image using Matplotlib.

    Parameters:
    - image (Union[torch.Tensor, np.ndarray]): Input image as a torch tensor or NumPy array.
    - window_name (str): Name of the window. Default is 'image'.
    """
    # Convert input to a NumPy array
    if isinstance(image, torch.Tensor):
        image_np = (image.numpy() * 255).astype(np.uint8)
    elif isinstance(image, np.ndarray):
        image_np = image
    else:
        raise ValueError("Unsupported image type. Accepted types: torch.Tensor, numpy.ndarray.")


    try:
        # Check if the image is grayscale or RGB
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:
            # RGB image
            plt.imshow(np.transpose(image_np, (1, 2, 0)))
        elif len(image_np.shape) == 2:
            # Grayscale image
            plt.imshow(image_np, cmap='gray')
        else:
            raise ValueError("Unsupported image format.")

        plt.show()
    except Exception as e:
        print(f"Error during display: {e}")

def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from the specified path.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - img (np.ndarray): Loaded image.
    """
    return cv.imread(image_path)


def compute_image_sum(image):
    """
    Compute the sum of pixel values in the input image.

    Parameters:
    - image (np.ndarray or torch.Tensor): Input image as a NumPy array or PyTorch tensor.

    Returns:
    - float: Sum of pixel values.
    """
    if isinstance(image, np.ndarray):
        return np.sum(image, axis=(0, 1), keepdims=True, dtype=np.float64)
    elif isinstance(image, torch.Tensor):
        return torch.sum(image).item()
    else:
        raise ValueError("Unsupported image type. Accepted types: np.ndarray, torch.Tensor.")

if __name__ == "__main__":

    pass
