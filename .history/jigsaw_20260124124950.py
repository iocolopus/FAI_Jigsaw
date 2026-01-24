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
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle),  np.cos(-angle)]])
        
        rotated = np.dot(self.contour - c1, R.T)
        
        tipo = self.kind
        
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


    @property
    def length(self):
        """Longitud de la arista (distancia euclidiana entre extremos)."""
        c1 = self.contour[0]
        c2 = self.contour[-1]
        return np.linalg.norm(c2 - c1)
    
    def straighten_contour_normalized(self, show=False):
        """
        Devuelve el contorno enderezado con el primer punto en (0, 0).
        No escala nada - mantiene las dimensiones reales.
        """
        c1 = self.contour[0]
        c2 = self.contour[-1]

        direction = c2 - c1
        angle = np.arctan2(direction[1], direction[0])
        R = np.array([[np.cos(-angle), -np.sin(-angle)],
                      [np.sin(-angle),  np.cos(-angle)]])
        
        rotated = np.dot(self.contour - c1, R.T)
        
        tipo = self.kind
        
        # Corregimos orientación según tipo
        if tipo == "macho" or tipo == "plano":
            rotated[:, 0] = -rotated[:, 0] + np.linalg.norm(direction)
        elif tipo == "hembra":
            rotated[:, 1] = -rotated[:, 1]
        
        if show:
            plt.plot(rotated[:, 0], rotated[:, 1])
            plt.plot(rotated[0, 0], rotated[0, 1], 'ro', label='inicio')
            plt.plot(rotated[-1, 0], rotated[-1, 1], 'go', label='fin')
            plt.axis('equal')
            plt.legend()
            plt.show()
            
        return rotated
    
    def dissimilarity(self, other, length_tolerance=10, show=False):
        """
        Calcula la disimilaridad entre dos aristas para ver si encajan.
        
        Solo tiene sentido comparar macho con hembra.
        
        Parámetros:
        - length_tolerance: diferencia máxima de longitud permitida (en píxeles)
        - show: si True, muestra gráficos de debug
        
        Retorna:
        - float: valor de disimilaridad (menor = mejor encaje)
        - float('inf') si no pueden encajar (tipos incompatibles o longitudes muy diferentes)
        """
        # 1. Verificar compatibilidad de tipos
        tipo_self = self.kind
        tipo_other = other.kind
        
        # Solo macho-hembra pueden encajar
        if not ((tipo_self == "macho" and tipo_other == "hembra") or 
                (tipo_self == "hembra" and tipo_other == "macho")):
            return float('inf')
        
        # 2. Verificar compatibilidad de longitudes (SIN ESCALAR)
        len_self = self.length
        len_other = other.length
        length_diff = abs(len_self - len_other)
        
        if length_diff > length_tolerance:
            if show:
                print(f"Longitudes incompatibles: {len_self:.1f} vs {len_other:.1f} (diff={length_diff:.1f})")
            return float('inf')
        
        # 3. Obtener contornos enderezados (sin escalar)
        contour1 = self.straighten_contour_normalized()
        contour2 = other.straighten_contour_normalized()
        
        # 4. Para comparar, necesitamos "voltear" uno de los contornos
        # porque cuando encajan, uno está espejado respecto al otro
        # Volteamos horizontalmente el segundo contorno
        contour2_flipped = contour2.copy()
        contour2_flipped[:, 0] = contour2[:, 0].max() - contour2_flipped[:, 0]
        contour2_flipped = contour2_flipped[::-1]  # invertir orden de puntos
        
        # 5. Interpolar ambos contornos para tener el mismo número de puntos
        n_points = 100
        contour1_interp = self._interpolate_contour(contour1, n_points)
        contour2_interp = self._interpolate_contour(contour2_flipped, n_points)
        
        # 6. Calcular área entre las curvas (disimilaridad)
        # Usamos la fórmula del área del polígono formado por ambas curvas
        polygon = np.vstack([contour1_interp, contour2_interp[::-1]])
        area = self._polygon_area(polygon)
        
        # También añadimos penalización por diferencia de longitud
        dissimilarity = area + length_diff * 2  # penalización proporcional
        
        if show:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot de los contornos superpuestos
            axes[0].plot(contour1_interp[:, 0], contour1_interp[:, 1], 'b-', label=f'self ({tipo_self})')
            axes[0].plot(contour2_interp[:, 0], contour2_interp[:, 1], 'r-', label=f'other ({tipo_other}) flipped')
            axes[0].fill_between(contour1_interp[:, 0], contour1_interp[:, 1], contour2_interp[:, 1], alpha=0.3, color='purple')
            axes[0].set_title(f'Disimilaridad: {dissimilarity:.1f}\nÁrea: {area:.1f}, ΔLongitud: {length_diff:.1f}')
            axes[0].legend()
            axes[0].axis('equal')
            
            # Plot de los contornos originales
            axes[1].plot(contour1[:, 0], contour1[:, 1], 'b-', label='self original')
            axes[1].plot(contour2[:, 0], contour2[:, 1], 'r-', label='other original')
            axes[1].set_title(f'Longitudes: {len_self:.1f} vs {len_other:.1f}')
            axes[1].legend()
            axes[1].axis('equal')
            
            plt.tight_layout()
            plt.show()
        
        return dissimilarity
    
    def _interpolate_contour(self, contour, n_points):
        """Interpola el contorno para tener n_points puntos equiespaciados."""
        # Calcular la longitud acumulada a lo largo del contorno
        diffs = np.diff(contour, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]
        
        # Crear puntos equiespaciados
        target_lengths = np.linspace(0, total_length, n_points)
        
        # Interpolar x e y
        x_interp = np.interp(target_lengths, cumulative_length, contour[:, 0])
        y_interp = np.interp(target_lengths, cumulative_length, contour[:, 1])
        
        return np.column_stack([x_interp, y_interp])
    
    def _polygon_area(self, vertices):
        """Calcula el área de un polígono usando la fórmula del shoelace."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return abs(area) / 2.0
        
        
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