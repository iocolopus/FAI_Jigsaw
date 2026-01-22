import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, index):
        back_path = "fotos/segmented_scan_25/back"
        front_path = "fotos/segmented_scan_25/front"

        self.front_img_path = f"{front_path}/piece_{index:02d}.png"

        self.index = index
        

    def display_back(self):
        path = self.

    def display_front(self):
        plt.
        




if __name__ == "__main__":
    df = pd.read_csv('esquinas_detectadas.csv')
    print(df.columns)