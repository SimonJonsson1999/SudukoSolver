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
from mnist import Net
from sudoku_solver import solve_sudoku

def preprocess_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the image and find contours of the Sudoku board.

    Parameters:
    - img (np.ndarray): Input image.

    Returns:
    - img_contours (np.ndarray): Image with contours drawn around the Sudoku board.
    - warped_board (np.ndarray): Perspective-transformed Sudoku board.
    """
    image_size = 252
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

    contours, _ = cv.findContours(img_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()

    largest_contour = max(contours, key=cv.contourArea)
    epsilon = 0.1 * cv.arcLength(largest_contour, True)
    approx = cv.approxPolyDP(largest_contour, epsilon, True)
    cv.drawContours(img_contours, [approx], -1, (0, 255, 0), 3)

    target_points = np.float32([[0, 0], [0, image_size], [image_size, 0], [image_size, image_size]])
    sorted_indices = np.lexsort((approx[:, 0][:, 1], approx[:, 0][:, 0]))
    ordered_points = approx[sorted_indices]

    ordered_points = np.float32(ordered_points)
    matrix = cv.getPerspectiveTransform(ordered_points, target_points)
    warped_board = cv.warpPerspective(img, matrix, (image_size, image_size))

    return img_contours.transpose(2,0,1), warped_board, ordered_points


def process_number_cell(cell: np.ndarray, size: Tuple = (28, 28)) -> torch.Tensor:
    """
    Process a single cell containing a number in the Sudoku board.

    Parameters:
    - cell (np.ndarray): Image of a Sudoku board cell.

    Returns:
    - t_number (torch.Tensor): Processed and normalized torch.Tensor representing the number.
    """
    gray_number = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    cropped_number = cv.resize(gray_number[4:-4, 4:-4], size, interpolation=cv.INTER_AREA)
    cropped_number = cropped_number[np.newaxis, :, :]
    # display_image(cropped_number.squeeze())
    threshold_value = 150
    max_value = 255
    _, binary_number = cv.threshold(cropped_number, threshold_value, max_value, cv.THRESH_BINARY)
    inverted_number = cv.bitwise_not(binary_number)

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    eroded_number = cv.dilate(inverted_number, kernel2, iterations=2)
    eroded_number = cv.erode(eroded_number, kernel1, iterations=1)
    


    t_number = torch.from_numpy(eroded_number).float()
    t_number /= 255.0
    return t_number


def get_board(image, correct_board = None) -> None:
    """
    Main function to execute the Sudoku board processing.
    """
    softmax = nn.Softmax(dim=1)

    wrong = 0
    board = np.zeros((9, 9), dtype=int)
    modelcnn = Net()
    modelcnn.load_state_dict(torch.load("mnist_cnn.pt"))
    modelcnn.eval()
    
    img_contours, warped_board, original_position = preprocess_image(image)
    cell_size = warped_board.shape[0] // 9
    cells = np.array([np.hsplit(row, 9) for row in np.vsplit(warped_board, 9)])
    for i in range(9):
        for j in range(9):
            number_cell = cells[i, j, :, :, :]
            t_number_rgb = process_number_cell(number_cell, size=(28, 28)).unsqueeze(0)
            if compute_image_sum(t_number_rgb[0]) > 50:
                output = modelcnn(t_number_rgb)
                probabilities = softmax(output)
                predictions = probabilities.argmax(dim=1)
            else:
                predictions = torch.tensor(0)
                # if type(correct_board):
                #     if predictions.item() != correct_board[i,j]:
                #         display_image(t_number_rgb.squeeze(0)[0])
                #         print(f"Prediction: {predictions.item()} Ground Truth: {correct_board[i,j]}, image sum = {compute_image_sum(t_number_rgb[0])}")
                #         wrong += 1
                # display_image(t_number_rgb[0])
            # print(f"Prediction: {predictions.item()} Ground Truth: {correct_board[i,j]}, image sum = {compute_image_sum(t_number_rgb[0])}")
            # display_image(t_number_rgb.squeeze(0)[0])

            board[i,j] = predictions.item()
    print(f"Number of wrongly classified cells: {wrong}, {100 * (wrong / 81):.2f}%")
    return board, original_position

    


if __name__ == "__main__":
    pass



