import numpy as np
import cv2 as cv
from functions import check_if_valid, draw_sudoku
N = 9 #size of board
sudoku =           np.array([    [0, 0, 1, 0, 4, 0, 0, 0, 6],
                        [0, 3, 5, 1, 0, 0, 0, 8, 0],
                        [0, 6, 0, 8, 0, 0, 0, 5, 7],
                        [5, 0, 0, 0, 0, 0, 0, 7, 0],
                        [6, 0, 0, 2, 0, 0, 0, 3, 1],
                        [8, 0, 0, 0, 1, 0, 5, 4, 9],
                        [9, 0, 6, 3, 0, 0, 0, 1, 0],
                        [0, 0, 0, 4, 0, 9, 0, 0, 0],
                        [0, 0, 7, 0, 6, 0, 0, 0, 0],])

def solve_sudoku(sudoku, row, col):

    if row == N - 1 and col == N:
        return True
      
    if col == N: 
        row += 1
        col = 0
    
    if sudoku[row][col] > 0:
        return solve_sudoku(sudoku, row, col + 1)
    
    for num in range(1,N+1):
        if check_if_valid(sudoku, row, col, num):
            sudoku[row][col] = num
            if solve_sudoku(sudoku, row, col + 1):
                return True
    
        sudoku[row][col] = 0
    return False


def main():

    if solve_sudoku(sudoku, 0, 0):
        print("-----|-Solved Board-|-----")
        print("- - - - - - - - - - - - ")
        draw_sudoku(sudoku)
    else:
        print("No solution exists!")
main()