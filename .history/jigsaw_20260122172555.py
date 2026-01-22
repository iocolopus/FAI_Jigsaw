import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, index):

        df = pd.read_csv('esquinas_detectadas.csv', converters={'contorno': eval, 'esquinas': eval})

        self.index = index
        self.contour = df['contorno'][index]
        self.corners = df['esquinas'][index]

        print(type(self.contour) + '\n' * 10)
        self._sort_corners()
        self.edges = self.generate_edges()

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

        edges = []

        for i in range(4):
            c1 = self.corners[i]
            c2 = self.corners[(i + 1) % 4]

            # Extraer el contorno entre c1 y c2
            i_c1 = np.where((self.contour == c1).all(axis=1))[0][0]
            i_c2 = np.where((self.contour == c2).all(axis=1))[0][0]

            if i_c1 < i_c2:
                edge_contour = self.contour[i_c1:i_c2 + 1]
            else:
                edge_contour = np.concatenate((self.contour[i_c1:], self.contour[:i_c2 + 1]), axis=0)

            edges.append(Edge(edge_contour))

        return edges
        
    


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

    def plot(self):
        plt.plot(self.contour[:, 0], self.contour[:, 1])
        plt.show()

    


class solver:
    def __init__(self):
        pass



def main():
    pieza = Piece(0)
    pieza.display_front()
    pieza.display_back()

    edge = pieza.edges[0]
    edge.plot()


if __name__ == "__main__":
    main()