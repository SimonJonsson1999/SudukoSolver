import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import cv2 as cv
import threading 
import time

from sudoku_solver import solve_sudoku
from get_sudoku import get_board
from functions import draw_sudoku
from utils import display_image, read_image, overlay_board


class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.geometry("1700x860")  
        self.root.configure(bg='#f0f0f0')  

        # Dictionary to hold original PIL images
        self.all_images_original = {
            "un_solved": None,
            "solved": None,
            "warped": None
        }

        # Dictionary to hold resized PhotoImage objects
        self.all_images_resized = {
            "un_solved": None,
            "solved": None,
            "warped": None
        }

        # Variables for storing the board
        self.board_image = None
        self.board = None
        self.original_board = None
        self.corners = None

        # Variables for the timer
        self.timer_running = False
        self.start_time = 0
        self.elapsed_time = 0
        self.timer_id = None

        # Spinner related variables
        self.spinner_angle = 0
        self.spinner_running = False

        # Style of buttons etc.
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 12))
        self.style.configure('TCombobox', font=('Helvetica', 12))

        self.create_widgets()
        self.load_images_from_folder("images")  

    def create_widgets(self):
        padding = 75

        # Frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        # Dropdown list
        self.image_combobox = ttk.Combobox(control_frame, width=50)
        self.image_combobox.grid(row=0, column=0, padx=25, pady=10, columnspan=2)

        # Load image button
        load_image_button = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        load_image_button.grid(row=0, column=2, padx=10, pady=10)

        # Solve Sudoku button
        solve_button = ttk.Button(control_frame, text="Solve", command=self.solve_board)
        solve_button.grid(row=0, column=3, padx=10, pady=10)

        # Timer label
        self.timer_label = ttk.Label(control_frame, text="Time: 0s")
        self.timer_label.grid(row=0, column=4, padx=10, pady=10)

        # Label to display scale percentage
        self.scale_label = ttk.Label(control_frame, text="Scale: 50%")
        self.scale_label.grid(row=1, column=5, padx=25, pady=5)

        # Scale to adjust image size
        self.scale = ttk.Scale(control_frame, from_=0.1, to=1, orient='horizontal', command=self.resize_images)
        self.scale.set(0.5)
        self.scale.grid(row=0, column=5, padx=25, pady=10)

        # Frame for displaying images
        image_frame = ttk.Frame(self.root)
        image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Make image_frame expandable
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Container frame for the Sudoku boards
        boards_frame = ttk.Frame(image_frame)
        boards_frame.grid(row=0, column=0, padx=75, pady=20, sticky="n")

        # Label to display un-solved sudoku
        self.un_solved = ttk.Label(boards_frame)
        self.un_solved.grid(row=0, column=0, padx=(0, 75), pady=20)

        # Label to display solved sudoku
        self.solved = ttk.Label(boards_frame)
        self.solved.grid(row=0, column=1, pady=20)
        self.solved.grid_remove()

        # Label to display warped sudoku
        self.warped = ttk.Label(boards_frame)
        self.warped.grid(row=1, column=0, padx=padding, pady=(20, 0), sticky="n")

        # Update position of the solved sudoku label
        self.solved.grid(row=0, column=1, rowspan=2, pady=20, sticky="n")

        # Update column configuration to accommodate the new label
        boards_frame.grid_columnconfigure(0, weight=1)
        boards_frame.grid_columnconfigure(1, weight=1)
        boards_frame.grid_columnconfigure(2, weight=1)

        # Canvas for the spinner
        self.spinner_canvas = tk.Canvas(boards_frame, width=100, height=100, bg='#f0f0f0', highlightthickness=0)
        self.spinner_canvas.grid(row=0, column=1, rowspan=2, pady=20, sticky="n")
        self.spinner_arc = self.spinner_canvas.create_arc((5, 5, 95, 95), start=0, extent=90, outline="blue", style="arc", width=5)
        self.spinner_canvas.grid_remove()

    def load_images_from_folder(self, folder):
        """
        Loads the images from a folder and add the path to the dropdown list
        """
        try:
            image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            self.image_combobox['values'] = image_files
            self.folder = folder
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load images from folder: {e}")
            print(f"Unable to load images from folder: {e}")

    def load_image(self):
        """
        Load the image selected from the dropdown list, try to get the board from the loaded image
        """
        selected_image = self.image_combobox.get()
        if selected_image:
            image_path = os.path.join(self.folder, selected_image)
            try:
                self.load_image_from_path(image_path)
                self.clear_solved_board()
                self.get_board_from_image(image_path)
                self.update_image_labels()  
            except Exception as e:
                messagebox.showerror("Error", f"Unable to load image: {e}")
                print(f"Unable to load image: {e}")
        else:
            messagebox.showwarning("Warning", "Please select an image from the dropdown")
            print("Warning: Please select an image from the dropdown")

    def solve_board(self):
        """
        Create a thread that solves the sudoku board
        """
        if self.board is not None:
            self.original_board = self.board.copy()
            self.start_timer()
            self.start_spinner()
            solve_thread = threading.Thread(target=self.solve_sudoku_in_thread)
            solve_thread.start()
        else:
            messagebox.showwarning("Warning", "Please load a board first")
            print("Warning: Please load a board first")

    def solve_sudoku_in_thread(self):
        """
        Solves and displays the sudoku
        """
        try:
            if solve_sudoku(self.board, 0, 0):
                self.stop_spinner()
                self.stop_timer()
                self.overlay_board()
                self.update_image_labels() 
            else:
                self.stop_spinner()
                self.stop_timer()
                messagebox.showerror("Error", "No solution exists for the selected board")
                print("No solution exists for the selected board")
        except Exception as e:
            self.stop_timer()
            messagebox.showerror("Error", f"Unable to solve board: {e}")
            print(f"Unable to solve board: {e}")


    def load_image_from_path(self, image_path):
        image = Image.open(image_path)
        self.un_solved_image = image
        self.all_images_original["un_solved"] = image

    def start_timer(self):
        self.start_time = time.time()
        self.elapsed_time = 0
        self.timer_running = True
        self.update_timer()

    def stop_timer(self):
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.timer_running = False

    def update_timer(self):
        if self.timer_running:
            self.elapsed_time = time.time() - self.start_time
            self.timer_label.config(text=f"Time: {self.elapsed_time:.3f}s")
            self.timer_id = self.root.after(100, self.update_timer)

    def start_spinner(self):
        self.spinner_running = True
        self.solved.grid_remove()  # Hide the solved board
        self.spinner_canvas.grid()  # Show the spinner
        self.update_spinner()

    def stop_spinner(self):
        self.spinner_running = False
        self.spinner_canvas.grid_remove()  # Hide the spinner
        self.solved.grid()  # Show the solved board

    def update_spinner(self):
        if self.spinner_running:
            self.spinner_angle = (self.spinner_angle + 10) % 360
            self.spinner_canvas.itemconfig(self.spinner_arc, start=self.spinner_angle)
            self.root.after(50, self.update_spinner)

    def resize_images(self, event=None):
        scale_factor = self.scale.get()
        for key, image in self.all_images_original.items():
            if image is not None:
                image_resized = image.resize(
                    (int(image.width * scale_factor), int(image.height * scale_factor)), Image.Resampling.LANCZOS)
                self.all_images_resized[key] = ImageTk.PhotoImage(image_resized)
                if key == "un_solved":
                    self.un_solved.config(image=self.all_images_resized[key])
                elif key == "solved":
                    self.solved.config(image=self.all_images_resized[key])
                elif key == "warped":
                    self.warped.config(image=self.all_images_resized[key])
        self.update_scale_label()

    def get_board_from_image(self,image_path):
        img_array = read_image(image_path)
        self.board_image = img_array
        self.board, self.corners, warped_board = get_board(img_array, debug=False)
        self.timer_label.config(text=f"Time: 0s")
        self.set_warped_board(warped_board)


    def overlay_board(self):
        overlayed_board = overlay_board(self.board_image, self.original_board, self.board, self.corners)
        overlayed_board = cv.cvtColor(overlayed_board, cv.COLOR_BGR2RGB)
        overlayed_image = Image.fromarray(overlayed_board)
        self.solved_image = overlayed_image
        self.all_images_original["solved"] = overlayed_image

    def update_image_labels(self):
        self.resize_images()

    def on_scale_change(self, event=None):
        self.update_scale_label()
        self.resize_images()

    def update_scale_label(self):
        scale_percentage = int(self.scale.get() * 100)
        self.scale_label.config(text=f"Scale: {scale_percentage}%")

    def set_warped_board(self, warped_board):
        warped_image = Image.fromarray(cv.cvtColor(warped_board, cv.COLOR_BGR2RGB))
        self.all_images_original["warped"] = warped_image

    def clear_solved_board(self):
        self.solved.config(image='')
        self.all_images_original["solved"] = None
        self.all_images_resized["solved"] = None
        self.warped.config(image='')
        self.all_images_original["warped"] = None
        self.all_images_resized["warped"] = None

                                             
if __name__ == "__main__":
    root = tk.Tk()
    gui = SudokuGUI(root)
    root.mainloop()
