import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, index):
        back_path = "fotos/segmented_scan_25/back"
        front_path = "fotos/segmented_scan_25/front"
        self.index = index
        

    def display_back(self):
        path = f"fotos/segmented_scan_25/back/piece_{self.index:02d}_back.png"

    def display_front(self):
        pass
        




if __name__ == "__main__":
    df = pd.read_csv('esquinas_detectadas.csv')
    print(df.columns)