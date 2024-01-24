import numpy as np
import torch
import torch.nn as nn
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
from utils import display_image, compute_image_sum, read_image


def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the image and find contours of the Sudoku board.

    Parameters:
    - img (np.ndarray): Input image.

    Returns:
    - img_contours (np.ndarray): Image with contours drawn around the Sudoku board.
    - warped_board (np.ndarray): Perspective-transformed Sudoku board.
    """
    
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
    trim_amount = 5
    cell = cell[trim_amount:-trim_amount, trim_amount:-trim_amount]
    # display_image(np.transpose(cell, (2, 0, 1)))
    gray_number = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    cropped_number = cv.resize(gray_number[4:-4, 4:-4], size, interpolation=cv.INTER_AREA)
    cropped_number = cropped_number[np.newaxis, :, :]
    
    threshold_value = 175
    max_value = 255
    _, binary_number = cv.threshold(cropped_number, threshold_value, max_value, cv.THRESH_BINARY)
    inverted_number = cv.bitwise_not(binary_number)
    
    kernel = np.ones((3, 3), np.uint8)
    eroded_number = cv.erode(inverted_number, kernel, iterations=3)
    
    t_number = torch.from_numpy(eroded_number)
    t_number_rgb = t_number.expand(1, 3, size[0], size[1]) / 255.0
    return t_number_rgb


def main() -> None:
    """
    Main function to execute the Sudoku board processing.
    """
    softmax = nn.Softmax(dim=1)
    correct_board = np.array([[0, 0, 2, 0, 7, 0, 0, 0, 5],
                        [0, 4, 3, 9, 5, 2, 0, 0, 0],
                        [0, 0, 0, 0, 6, 0, 0, 2, 4],
                        [0, 0, 0, 3, 1, 0, 8, 0, 0],
                        [4, 0, 0, 0, 2, 0, 0, 0, 6],
                        [0, 0, 1, 0, 0, 7, 0, 0, 0],
                        [5, 9, 0, 0, 4, 0, 0, 0, 0],
                        [0, 0, 4, 0, 0, 1, 7, 5, 0],
                        [2, 0, 0, 0, 3, 0, 4, 0, 0],])
    wrong = 0
    wrong_index = np.zeros((9,9))
    board = np.zeros((9, 9), dtype=int)
    model_name = "farleyknight-org-username/vit-base-mnist"
    model = ViTForImageClassification.from_pretrained(model_name)
    image_path = "sudoku.jpg"
    image = read_image(image_path)
    img_contours, warped_board = preprocess_image(image)

    # Example processing of a cell containing a number
    cell_size = warped_board.shape[0] // 9
    cells = np.array([np.hsplit(row, 9) for row in np.vsplit(warped_board, 9)])

    for i in range(9):
        for j in range(9):
            number_cell = cells[i, j, :, :, :]
            t_number_rgb = process_number_cell(number_cell, size=(224, 224))
            if compute_image_sum(t_number_rgb[0]) > 8_000:
                # print(t_number_rgb.shape)
                outputs = model(pixel_values=t_number_rgb)
                logits = outputs.logits[:, 1:]
                
                probabilities = softmax(logits)
                # print(f"Probabilities: {', '.join([f'{digit}:{prob:.4f}' for digit, prob in enumerate(probabilities.squeeze(), 1)])}")
                predictions = probabilities.argmax(dim=1) + 1
            else:
                predictions = torch.tensor(0)
            # print(f"Prediction: {predictions.item()} Ground Truth: {correct_board[i,j]}, image sum = {compute_image_sum(t_number_rgb[0])}")
            if predictions.item() != correct_board[i,j]:
                print(f"Prediction: {predictions.item()} Ground Truth: {correct_board[i,j]}, image sum = {compute_image_sum(t_number_rgb[0])}")
                print(f"Probabilities: {', '.join([f'{digit}:{prob:.4f}' for digit, prob in enumerate(probabilities.squeeze(), 1)])}")
                wrong += 1
                wrong_index[i,j] = 1
                display_image(t_number_rgb[0])

            # display_image(t_number_rgb[0])

            board[i,j] = predictions.item()
    print(f"Number of wrongly classified cells: {wrong}, {100 * (wrong / 81):.2f}%")

    draw_sudoku(board)
    


if __name__ == "__main__":
    main()



