import document_scanner
import Solver
import tensorflow as tf
import numpy as np

MODEL_PATH = 'save_model\digit_recognizer_300'
FILE_PATH = 'v2_test/image83.jpg'


def extract_solve(model_path, file_path, show_extract=False):
    load_model = tf.keras.models.load_model(model_path)
    board = document_scanner.generate_board_df(file_path)
    board_raw = load_model(board.values)
    board_clean = np.argmax(board_raw, axis=1).reshape((9, 9))
    if show_extract:
        print(board_clean)
    Solver.run_solver(board_clean)


extract_solve(MODEL_PATH, FILE_PATH)