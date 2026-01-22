import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, index):


        self.index = index

       # Asociamos a cada pieza su imagen por delante y por detras
        back_path = "fotos/segmented_scan_25/back"
        front_path = "fotos/segmented_scan_25/front"
        self.front_img_path = f"{front_path}/{index:02d}.png"
        self.back_img_path = f"{back_path}/{index:02d}.png"

        
        

    def display_back(self):
        plt.imshow(cv.cvtColor(cv.imread(self.back_img_path), cv.COLOR_BGR2RGB))
        plt.show()

    def display_front(self):
        plt.imshow(cv.cvtColor(cv.imread(self.front_img_path), cv.COLOR_BGR2RGB))
        




if __name__ == "__main__":
    df = pd.read_csv('esquinas_detectadas.csv')
    print(df.columns)