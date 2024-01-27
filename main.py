import numpy as np

from utils import display_image, compute_image_sum, read_image, overlay_board
from get_sudoku import get_board
from functions import draw_sudoku
from sudoku_solver import solve_sudoku




def main():
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
    
    image_path = "sudoku3.jpg"
    image_path = "sudoku.png"
    test_image = "test.png"
    image = read_image(image_path)
    print(f"image size: {image.shape}")
    board, board_corners  = get_board(image)
    original_board = board.copy()
    print(board_corners )
    print("-----|-Start Board-|-----")
    print("- - - - - - - - - - - - ")
    draw_sudoku(board)
    if solve_sudoku(board, 0, 0):
        print("-----|-Solved Board-|-----")
        print("- - - - - - - - - - - - ")
        draw_sudoku(board)
        overlayed_board = overlay_board(image, original_board, board, board_corners)
        display_image(overlayed_board.transpose(2,0,1))

if __name__ == "__main__":
    main()