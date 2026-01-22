import pandas as pd
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, ):
        self.corners = corners  # Expecting a list of (x, y) tuples

    def display(self):
        for corner in self.corners:
            print(f"Corner at: {corner}")




if __name__ == "__main__":
    df = pd.read_csv('esquinas_detectadas.csv')
    df.dtypes