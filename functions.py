import numpy as np
N = 9
def check_if_in_row(sudoku, row, num):
    for col in range(N):
        if sudoku[row][col] == num:
            return True
    return False


def check_if_in_col(sudoku, col, num):
    for row in range(N):
        if sudoku[row][col] == num:
            return True
    return False

def check_if_in_box(sudoku, start_row, start_col, num):
    box_start_row = start_row - start_row % 3
    box_start_col = start_col - start_col % 3
    for row in range(3):
        for col in range(3):
            if sudoku[row + box_start_row][col + box_start_col] == num:
                return True
    return False

def check_if_valid(sudoku, row,  col, num):
    return not check_if_in_row(sudoku, row, num) and not check_if_in_col(sudoku, col, num) and not check_if_in_box(sudoku, row, col, num)

def draw_sudoku(sudoku):
    print("- - - - - - - - - - - - ")
    for row in range(len(sudoku)):
            if row % 3 == 0 and row != 0:
                print("- - - - - - - - - - - - ")
            for col in range(len(sudoku[0])):
                if col % 3 == 0 and col != 0:
                    print(" | ", end="")

                if col == 8:
                    print(sudoku[row][col])
                else:
                    print(str(sudoku[row][col]) + " ", end="")
    print("- - - - - - - - - - - - ")

def predict(img, model):
    prediction = model.predict(img)
    return np.argmax(prediction)

    