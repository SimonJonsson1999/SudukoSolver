import numpy as np
import time
import argparse
from utils import display_image, read_image, overlay_board
from get_sudoku import get_board
from functions import draw_sudoku
from sudoku_solver import solve_sudoku




def main(debug=False):
    correct_board1 = np.array([[0, 0, 2, 0, 7, 0, 0, 0, 5],
                        [0, 4, 3, 9, 5, 2, 0, 0, 0],
                        [0, 0, 0, 0, 6, 0, 0, 2, 4],
                        [0, 0, 0, 3, 1, 0, 8, 0, 0],
                        [4, 0, 0, 0, 2, 0, 0, 0, 6],
                        [0, 0, 1, 0, 0, 7, 0, 0, 0],
                        [5, 9, 0, 0, 4, 0, 0, 0, 0],
                        [0, 0, 4, 0, 0, 1, 7, 5, 0],
                        [2, 0, 0, 0, 3, 0, 4, 0, 0],])
    
    correct_board2 = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
                                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                                [0, 0, 0, 0, 8, 0, 0, 7, 9],])
    
    image_path1 = r"images\sudoku.png"
    image_path2 = r"images\sudoku2.png"
    image_path3 = r"images\sudoku3.jpg"
    image = read_image(image_path1)
    print(f"image size: {image.shape}")
    board, board_corners, warped_board  = get_board(image, debug)
    original_board = board.copy()
    print("-----|-Start Board-|-----")
    print("- - - - - - - - - - - - ")
    draw_sudoku(board)
    start_time = time.time()
    if solve_sudoku(board, 0, 0):
        end_time = time.time()
        print("-----|-Solved Board-|-----")
        print("- - - - - - - - - - - - ")
        draw_sudoku(board)
        print(type(image))
        print(type(original_board))
        print(type(board))
        print(type(board_corners))
        overlayed_board = overlay_board(image, original_board, board, board_corners)
        print(type(overlay_board))
        display_image(overlayed_board.transpose(2,0,1))
        print(f"Sudoku solved in {end_time - start_time:.6f} seconds")
    else:
        print("No solution exists")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sudoku Solver')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    main(debug=args.debug)