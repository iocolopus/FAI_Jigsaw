import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.interpolate import CubicSpline, interp1d

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

    @property
    def kind(self):
        """Clasificamos cada tipo de arista segun sin es entrante o saliente, de modo que solo medimos la similaridad entre aristas de distinto tipo puesto que son las que en cualquier caso encajarian."""

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

    @property
    def straighten_contour(self):
        """Hacemos que el contorno este recto para poder hacer las comparaciones entre aristas independientemente de como estuviera cada arista orientada en la foto."""

        # sea todas pa arriba
        c1 = self.contour[0]
        c2 = self.contour[-1]

        direction = c2 - c1
        angle = np.arctan2(direction[1], direction[0])
        # Prueba rara

        kind = self.kind

        if kind == 'hembra':
            R = np.array([[np.cos(np.pi - angle), -np.sin(np.pi - angle)],
                          [np.sin(np.pi - angle),  np.cos(np.pi - angle)]])
            
            rotated = np.dot(self.contour - c2, R.T)
        else:
            R = np.array([[np.cos(-angle), -np.sin(-angle)],
                          [np.sin(-angle),  np.cos(-angle)]])
        
            rotated = np.dot(self.contour - c1, R.T)
            # Invertir el orden en el que se recorren los puntos para que siempre vaya de izquierda a derecha
            rotated = rotated[]
        

        return rotated


    @property
    def length(self):
        """Longitud de la arista (distancia euclidiana entre extremos)."""
        c1 = self.contour[0]
        c2 = self.contour[-1]
        return np.linalg.norm(c2 - c1)
    

    
    def abs_len_diff(self, other):
        return abs(self.length - other.length)
    
    def resample_contour_uniform(self, n_samples=50, plot=False):
        """
        Remuestra el contorno de la arista de forma uniforme usando splines cúbicos parametrizados por longitud de arco.
        
        Args:
            n_samples: Número de puntos a generar (default 200)
            
        Returns:
            Tuple de (x_resampled, y_resampled) con n_samples puntos cada uno, uniformemente distribuidos en longitud de arco.
        """
        contour = self.straighten_contour.astype(np.float32)
        
        # Calcular longitudes de arco acumuladas
        diffs = np.diff(contour, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
        
        # Extraer coordenadas x e y
        x_orig = contour[:, 0]
        y_orig = contour[:, 1]
        
        # Crear splines cúbicos parametrizados por longitud de arco
        # s es el parámetro (longitud de arco), x(s) e y(s) son funciones suaves
        spline_x = interp1d(arc_lengths, x_orig, kind='linear')
        spline_y = interp1d(arc_lengths, y_orig, kind='linear')
        
        # Generar n_samples puntos uniformemente distribuidos en la longitud de arco
        s_resampled = np.linspace(0, arc_lengths[-1], n_samples)
        
        # Evaluar los splines en los nuevos puntos de parámetro
        x_resampled = spline_x(s_resampled)
        y_resampled = spline_y(s_resampled)

        if plot:
            plt.plot(x_orig, y_orig, '-', label='Original')
            plt.plot(x_resampled, y_resampled, 'o', label='Remuestreado', color='green')
            plt.axis('equal')
            plt.legend()
            plt.show()
        
        return x_resampled, y_resampled
    
    def dissimilarity(self, other, plot=False):
        """
        Calcula una medida de disimilitud entre esta arista y otra.
        Utiliza remuestreo uniforme y suma de diferencias cuadráticas.
        
        Args:
            other: Otra instancia de Edge para comparar.
            show: Si es True, muestra los contornos remuestreados.
            
        Returns:
            Valor numérico que representa la disimilitud entre las dos aristas.
        """
        n_samples = 50
        
        x1, y1 = self.resample_contour_uniform(n_samples=n_samples)
        x2, y2 = other.resample_contour_uniform(n_samples=n_samples)
        
        # Calcular disimilitud como suma de diferencias cuadráticas
        dissimilarity_value = np.sum((x1 - x2) ** 2 + (y1 - y2) ** 2)

        if plot:
            plt.plot(x1, y1, 'o-', label='Arista 1')
            plt.plot(x2, y2, 'o-', label='Arista 2')
            plt.title(f'Dissimilarity: {dissimilarity_value:.2f}')
            plt.axis('equal')
            plt.legend()
            plt.show()

        # ploteamos un segmenteo de recta entre cada par de puntos para ver mejor la diferencia
        if plot:
            for i in range(n_samples):
                plt.plot([x1[i], x2[i]], [y1[i], y2[i]], 'r-')
            plt.title('Diferencias entre puntos remuestreados')
            plt.axis('equal')
            plt.show()
        
        return dissimilarity_value
    
        
        
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