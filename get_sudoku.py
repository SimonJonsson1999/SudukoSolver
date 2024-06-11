import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib as plt
from torchvision import datasets, transforms
from PIL import Image

import cv2 as cv
import numpy as np
import torch
from typing import Tuple

from functions import draw_sudoku
from utils import display_image, compute_image_sum, read_image
from mnist import Net
from sudoku_solver import solve_sudoku

def order_points(pts):
    """
    Order points in the following order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the image and find contours of the Sudoku board.

    Parameters:
    - img (np.ndarray): Input image.

    Returns:
    - img_contours (np.ndarray): Image with contours drawn around the Sudoku board.
    - warped_board (np.ndarray): Perspective-transformed Sudoku board.
    - ordered_points (np.ndarray): Points of the detected Sudoku board.
    """
    image_size = 252
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv.findContours(img_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()

    largest_contour = max(contours, key=cv.contourArea)
    epsilon = 0.02 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        print(len(approx))
        raise ValueError("Could not find a proper quadrilateral for the Sudoku board.")

    approx = approx.reshape((4, 2))
    ordered_points = order_points(approx)

    cv.drawContours(img_contours, [ordered_points.astype(int)], -1, (0, 255, 0), 3)

    target_points = np.float32([[0, 0], [image_size, 0], [image_size, image_size], [0, image_size]])
    matrix = cv.getPerspectiveTransform(ordered_points, target_points)
    warped_board = cv.warpPerspective(img, matrix, (image_size, image_size))

    return img_contours, warped_board, ordered_points


def process_number_cell(cell: np.ndarray, size: Tuple = (28, 28)) -> torch.Tensor:
    """
    Process a single cell containing a number in the Sudoku board.

    Parameters:
    - cell (np.ndarray): Image of a Sudoku board cell.

    Returns:
    - t_number (torch.Tensor): Processed and normalized torch.Tensor representing the number.
    """
    transform=transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    crop = 3
    threshold = 70
    gray_number = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    
    cropped_number = cv.resize(gray_number[crop:-crop, crop:-crop], size, interpolation=cv.INTER_CUBIC )
    if cropped_number.ndim == 3:
        print(f"Converting 3-channel image to single-channel: {cropped_number.shape}")
        cropped_number = cropped_number.squeeze()
    
    # display_image(cropped_number.squeeze())
    
    inverted_number = cv.bitwise_not(cropped_number)
    inverted_number[inverted_number < threshold] = 1
    # display_image(inverted_number.squeeze())
    inverted_number = inverted_number[np.newaxis, :, :]
    
    t_number = transform(torch.from_numpy(inverted_number).float())
    return t_number


def get_board(image, correct_board = None) -> None:
    """
    Main function to execute the Sudoku board processing.
    """
    softmax = nn.Softmax(dim=1)
    wrong = 0
    board = np.zeros((9, 9), dtype=int)
    modelcnn = Net()
    modelcnn.load_state_dict(torch.load("models/mnist_cnn.pt"))
    modelcnn.eval()

    img_contours, warped_board, original_position = preprocess_image(image)
    cell_size = warped_board.shape[0] // 9
    cells = np.array([np.hsplit(row, 9) for row in np.vsplit(warped_board, 9)])
    for i in range(9):
        for j in range(9):
            number_cell = cells[i, j, :, :, :]
            t_number_rgb = process_number_cell(number_cell, size=(28, 28)).unsqueeze(0)
            print(compute_image_sum(t_number_rgb[0]))
            if compute_image_sum(t_number_rgb[0]) > 9_000:
                output = modelcnn(t_number_rgb)
                probabilities = softmax(output)
                predictions = probabilities.argmax(dim=1)
                print(predictions.item())
                board[i,j] = predictions.item()
            else:
                predictions = torch.tensor(0)
            
    return board, original_position

    


if __name__ == "__main__":
    pass



