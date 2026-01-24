import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Piece:
    def __init__(self, index):

        df = pd.read_pickle('esquinas_detectadas.pkl')

        self.index = index
        self.contour = df['contorno'][index]
        self.corners = df['esquinas'][index]

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
        # Genera los objetos aristas la arista 0 er arriba y continua aumentando en el sentido de las aguas del reloj.

        edges = []

        for i in range(4):
            c2 = self.corners[i]
            c1 = self.corners[(i + 1) % 4]

            # Extraer el contorno entre c1 y c2
            i_c1 = np.where((self.contour == c1).all(axis=1))[0][0]
            i_c2 = np.where((self.contour == c2).all(axis=1))[0][0]

            if i_c1 < i_c2:
                edge_contour = self.contour[i_c1:i_c2 + 1]
            else:
                edge_contour = np.concatenate((self.contour[i_c1:], self.contour[:i_c2 + 1]), axis=0)

            edges.append(Edge(edge_contour, self))

        return edges
        
    


    def display_back(self):
        plt.imshow(cv.cvtColor(cv.imread(self.back_img_path), cv.COLOR_BGR2RGB))
        plt.show()

    def display_front(self):
        plt.imshow(cv.cvtColor(cv.imread(self.front_img_path), cv.COLOR_BGR2RGB))
        plt.show()



    

class Edge():

    def __init__(self, contour, pieza):
        self.contour = contour
        self.pieza = pieza

    def kind(self):
        # 0 plano 1 macho 2 hembra
        
        c1 = self.contour[0]
        c2 = self.contour[-1]
        
        # Punto medio del conjunto de puntos del borde.
        middle_point_edge = self.contour[len(self.contour)//2]
        
        # Punto medio de la recta entre los extremos del borde.
        center_line = (c1 + c2) / 2
        
        # Vector desde el centro de la línea al punto medio del borde
        vector_middle_point_to_center_line = middle_point_edge - center_line
        distance_middle_point_to_center_line = np.linalg.norm(vector_middle_point_to_center_line)
        
        # Si la distancia entre los dos puntos es menor que 5 significa que están muy cerca, lo consideramos plano.
        if distance_middle_point_to_center_line < 5:
            return 'plano'
        
        # Calculamos centroide de la pieza y vector desde el centro de la línea al centroide
        centroide = np.mean(self.pieza.contour, axis=0)
        vector_center_line_to_centroid = centroide - center_line
        
        # Si la distancia del centro de la línea al centro del borde y la distancia del centro de la línea al centroide es la misma
        # significa que es entrante. Si van en direcciones opuestas, es que el centro del borde está en dirección opuesta al centroide desde el
        # centro de la línea.
        producto_punto = np.dot(vector_middle_point_to_center_line, vector_center_line_to_centroid)
        if producto_punto > 0:
            return 'hembra'
        else:
            return 'macho'

    def straighten_contour(self):
        # sea todas pa arriba
        c1 = self.contour[0]
        c2 = self.contour[-1]

        direction = c2 - c1
        angle = np.arctan2(direction[1], direction[0])
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle),  np.cos(-angle)]])
        
        rotated = np.dot(self.contour - c1, R.T)
        
        tipo = self.kind()
        
        # con la rotación que habíamos aplicado, se hacía como un flip horizontal
        # y quedaba al revés, con esto simplemente lo corregimos para cada caso para que quede
        # todo hacia arriba
        if tipo == "macho" or tipo == "plano":
            rotated[:, 0] = -rotated[:, 0] + np.linalg.norm(direction)
        elif tipo == "hembra":
            rotated[:, 1] = -rotated[:, 1]
        
        rotated[:,1] += 100
        
        plt.plot(rotated[:, 0], rotated[:, 1])
        plt.plot(rotated[0, 0], rotated[0, 1], 'ro')
        plt.plot(rotated[-1, 0], rotated[-1, 1], 'go')
        
        plt.xlim(-10,280)
        plt.ylim(0,200)
        plt.show()
        return rotated

    def masked_contour(self):
        mask = np.zeros((200, 280), dtype=np.uint8)
        straightened = self.straighten_contour()
        tipo = self.kind()
        c1, c2 = straightened[0], straightened[-1]
        
        if tipo == "macho" or tipo == "plano":
            # con esto creo 4 puntos más para cerrar el polígono
            straightened = np.vstack([[280, c1[1]], straightened, [0, c2[1]], [0,0], [280,0]])
        if tipo == "hembra":
            # aquí más de lo mismo
            straightened = np.vstack([[0, c1[1]], straightened, [280, c2[1]], [280,280], [0,280]])
        
        cv.fillPoly(mask, [straightened.astype(np.int32)], 255)
        
        plt.imshow(mask, cmap='gray', origin='lower')
        plt.show()
        
        return mask
    
    def xor_masked_contour(self, other):
        mask1 = self.masked_contour()
        mask2 = other.masked_contour()
        
        xor_mask = cv.bitwise_xor(mask1, mask2)
        
        plt.imshow(xor_mask, cmap='gray', origin='lower')
        plt.show()
        
        return xor_mask
        
        
    def plot(self):
        plt.imshow(cv.cvtColor(cv.imread(self.pieza.back_img_path), cv.COLOR_BGR2RGB))
        plt.plot(self.contour[:, 0], self.contour[:, 1])
        c1 = self.contour[0]
        c2 = self.contour[-1]
        plt.plot(c1[0], c1[1], 'ro')
        plt.plot(c2[0], c2[1], 'go')
        plt.show()

    def __sub__(self, other):
        # medida de similaridad
        pass
        

    


class solver:
    def __init__(self):
        pass



def main():
    pieza = Piece(14)
    pieza.display_front()
    pieza.display_back()

    edge = pieza.edges[0]
    edge.plot()


if __name__ == "__main__":
    main()