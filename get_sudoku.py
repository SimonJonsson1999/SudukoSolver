import numpy as np
import torch
import cv2 as cv
import matplotlib as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

import cv2 as cv
import numpy as np
import torch
from typing import Tuple

from functions import draw_sudoku
from utils import display_image, compute_image_sum


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read, preprocess, and find contours of the Sudoku board in an image.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - img_contours (np.ndarray): Image with contours drawn around the Sudoku board.
    - warped_board (np.ndarray): Perspective-transformed Sudoku board.
    """
    img = cv.imread(image_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

    contours, _ = cv.findContours(img_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()

    largest_contour = max(contours, key=cv.contourArea)
    epsilon = 0.1 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)
    cv.drawContours(img_contours, [approx], -1, (0, 255, 0), 3)

    target_points = np.float32([[0, 0], [0, 450], [450, 450], [450, 0]])
    ordered_points = np.array([approx[1, 0], approx[2, 0], approx[3, 0], approx[0, 0]], dtype=np.float32)

    matrix = cv.getPerspectiveTransform(ordered_points, target_points)
    warped_board = cv.warpPerspective(img, matrix, (450, 450))

    return img_contours, warped_board


def process_number_cell(cell: np.ndarray, size: Tuple = (28, 28)) -> torch.Tensor:
    """
    Process a single cell containing a number in the Sudoku board.

    Parameters:
    - cell (np.ndarray): Image of a Sudoku board cell.

    Returns:
    - t_number (torch.Tensor): Processed and normalized torch.Tensor representing the number.
    """
    number = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    number = cv.resize(number[4:-4, 4:-4], size, interpolation=cv.INTER_AREA)
    number = number[np.newaxis, :, :]
    display_image(number)
    threshold_value = 175
    max_value = 255
    _, t_number = cv.threshold(number, threshold_value, max_value, cv.THRESH_BINARY)
    t_number = cv.bitwise_not(t_number)
    kernel = np.ones((2, 2), np.uint8)
    t_number = cv.erode(t_number, kernel, iterations=1)
    t_number = torch.from_numpy(t_number)

    t_number_rgb = t_number.expand(1, 3, size[0], size[1])
    t_number_rgb = t_number_rgb / 255.0

    return t_number_rgb


def main() -> None:
    """
    Main function to execute the Sudoku board processing.
    """
    board = np.zeros((9, 9), dtype=int)
    model_name = "farleyknight-org-username/vit-base-mnist"
    model = ViTForImageClassification.from_pretrained(model_name)
    image_path = "sudoku.jpg"
    img_contours, warped_board = preprocess_image(image_path)

    # Example processing of a cell containing a number
    cell_size = warped_board.shape[0] // 9
    cells = np.array([np.hsplit(row, 9) for row in np.vsplit(warped_board, 9)])

    for i in range(9):
        for j in range(9):
            number_cell = cells[i, j, :, :, :]
            t_number_rgb = process_number_cell(number_cell, size=(224, 224))
            if compute_image_sum(t_number_rgb[0]) > 8_000:

                outputs = model(pixel_values=t_number_rgb)
                predictions = outputs.logits.argmax(dim=1)
            else:
                predictions = torch.tensor(0)
            print(f"Prediction: {predictions.item()}, image sum = {compute_image_sum(t_number_rgb[0])}")
            display_image(t_number_rgb[0])
            board[i,j] = predictions.item()

    draw_sudoku(board)
    


if __name__ == "__main__":
    main()



