import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, index):


        self.index = index
        self.contour = df['contorno'][index]
        self.corners = df['esquinas'][index]
        self._sort_corners()

       # Asociamos a cada pieza su imagen por delante y por detras
        back_path = "fotos/segmented_scan_25/back"
        front_path = "fotos/segmented_scan_25/front"
        self.front_img_path = f"{front_path}/{index:02d}.png"
        self.back_img_path = f"{back_path}/{index:02d}.png"

    def _sort_corners(self):
        def angle(corner):
            centroide = np.mean(self.corners, axis=0)
            vector = corner - centroide
            return np.arctan2(vector[1], vector[0])

        self.corners = sorted(self.corners, key=angle)


    def generate_edges(self):

        for i in range(5):
            start_corner = self.corners[i]
            end_corner = self.corners[(i + 1) % 4]
            mask = np.zeros((500, 500), dtype=np.uint8)
            cv.drawContours(mask, [self.contour], -1, 255, -1)
            cv.line(mask, tuple(start_corner), tuple(end_corner), 255, 5)
            edge_contour = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
            edge = Edge(edge_contour)
            # Aquí podríamos almacenar el edge en una lista si es necesario

    


    def display_back(self):
        plt.imshow(cv.cvtColor(cv.imread(self.back_img_path), cv.COLOR_BGR2RGB))
        plt.show()

    def display_front(self):
        plt.imshow(cv.cvtColor(cv.imread(self.front_img_path), cv.COLOR_BGR2RGB))
        plt.show()

    def get_edges(self):
        edges = [None] * 4



    

class Edge():

    def __init__(self, contour):
        self.contour = contour


class solver:
    def __init__(self):
        pass



def main():
    global df
    df = pd.read_csv('esquinas_detectadas.csv')
    pieza = Piece(0)
    pieza.display_front()
    pieza.display_back()


if __name__ == "__main__":
    main()