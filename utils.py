import numpy as np
import torch
from typing import Union
import matplotlib.pyplot as plt
import cv2 as cv
from IPython.display import display, Image

import torch
from torchvision import transforms, models
from model import MNistNet


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

def display_image_notebook(image: Union[torch.Tensor, np.ndarray], window_name: str = 'image') -> None:
    """
    Display an image using Matplotlib in a Jupyter Notebook.

    Parameters:
    - image (Union[torch.Tensor, np.ndarray]): Input image as a torch tensor or NumPy array.
    - window_name (str): Name of the window. Default is 'image'.
    """
    if isinstance(image, torch.Tensor):
        image_np = (image.numpy() * 255).astype(np.uint8)
    elif isinstance(image, np.ndarray):
        image_np = image
    else:
        raise ValueError("Unsupported image type. Accepted types: torch.Tensor, numpy.ndarray.")

    try:
        # Check if the image is grayscale or RGB
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # RGB image
            display(Image.fromarray(image_np))
        elif len(image_np.shape) == 2:
            # Grayscale image
            display(Image.fromarray(image_np, 'L'))
        else:
            raise ValueError("Unsupported image format.")
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

def load_model(model_path):
    backbone = models.resnet18(weights=None)  
    model = MNistNet(backbone)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model


def overlay_board(image, original_board, solved_board, board_corners):
    # print(board_corners)
    cell_width = (board_corners[1][0] - board_corners[0][0]) / 9
    cell_height = (board_corners[2][1] - board_corners[0][1]) / 9

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    offset = 10

    for i in range(9):
        for j in range(9):
            if original_board[i, j] == 0 and solved_board[i, j] != 0:
                x = int(board_corners[0][0] + j * cell_width + cell_width / 2 - offset)
                y = int(board_corners[0][1] + i * cell_height + cell_height / 2 + offset)
                cv.putText(image, str(int(solved_board[i, j])), (x, y),
                           font, font_scale, (23, 111, 32), font_thickness, cv.LINE_AA)
    return image
if __name__ == "__main__":

    pass
