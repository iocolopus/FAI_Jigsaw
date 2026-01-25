import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.interpolate import CubicSpline, interp1d
from IPython.display import display, HTML, clear_output

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
        self.kind = self.clasify_kind()  


    def clasify_kind(self):
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
            rotated = rotated[::-1]
        

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
        Remuestra el contorno de la arista de forma uniforme usando splines parametrizados por longitud de arco.
        
        Args:
            n_samples: Número de puntos a generar (default 50)
            
        Returns:
            Tuple de (x_resampled, y_resampled)
        """
        contour = self.straighten_contour.astype(np.float32)
        
        # Calcular longitudes de arco acumuladas
        diffs = np.diff(contour, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
        
        # Extraer coordenadas x e y
        x_orig = contour[:, 0]
        y_orig = contour[:, 1]
        
        # Crear splines parametrizados por longitud de arco
        spline_x = interp1d(arc_lengths, x_orig, kind='linear')
        spline_y = interp1d(arc_lengths, y_orig, kind='linear')
        
        # Generar n_samples puntos uniformemente distribuidos en la longitud de arco
        s_resampled = np.linspace(0, arc_lengths[-1], n_samples)
        
        # Evaluar los splines
        x_resampled = spline_x(s_resampled)
        y_resampled = spline_y(s_resampled)

        if plot:
            plt.plot(x_orig, y_orig, '-', label='Original')
            plt.plot(x_resampled, y_resampled, 'o', label='Remuestreado', color='green')
            plt.axis('equal')
            plt.legend()
            plt.show()
        
        return x_resampled, y_resampled

    def dissimilarity(self, other, c1 = 0, n_samples=50, plot=False):
        # c1: regula la importancia que se le da en la disimilaridad la diferencia de longitudes entre aristas
        """
        Calcula una medida de disimilitud entre esta arista y otra.
        Alinea desde (0,0) y hace ajuste fino para minimizar la distancia.
        
        Args:
            other: Otra instancia de Edge para comparar.
            c1: Coeficiente que regula la importancia de la diferencia de longitudes entre aristas.
            
        Returns:
            Valor numérico que representa la disimilitud entre las dos aristas.
            None si los kinds son incompatibles (iguales o alguno es plano).
        """

        # Si los kinds son iguales o alguno es plano, no son compatibles
        if self.kind == other.kind or self.kind == 'plano' or other.kind == 'plano':
            return np.inf
        
        x1, y1 = self.resample_contour_uniform(n_samples=n_samples)
        x2, y2 = other.resample_contour_uniform(n_samples=n_samples)
        
        # Alinear ambos contornos desde (0, 0) - el primer punto en el origen
        x1 = x1 - x1[0]
        y1 = y1 - y1[0]
        x2 = x2 - x2[0]
        y2 = y2 - y2[0]
        
        # Ajuste fino: buscar la traslación (tx, ty) que minimiza el error
        best_error = np.inf
        best_tx, best_ty = 0, 0
        
        # Búsqueda en un rango amplio
        for tx in np.arange(-20, 20.5, 1):
            for ty in np.arange(-20, 20.5, 1):
                error = np.sum((x1 - (x2 + tx)) ** 2 + (y1 - (y2 + ty)) ** 2)
                if error < best_error:
                    best_error = error
                    best_tx, best_ty = tx, ty
        
        # Refinamiento fino
        for tx in np.arange(best_tx - 1, best_tx + 1.1, 0.1):
            for ty in np.arange(best_ty - 1, best_ty + 1.1, 0.1):
                error = np.sum((x1 - (x2 + tx)) ** 2 + (y1 - (y2 + ty)) ** 2)
                if error < best_error:
                    best_error = error
                    best_tx, best_ty = tx, ty
        
        # Aplicar la mejor traslación
        x2 = x2 + best_tx
        y2 = y2 + best_ty
        
        # Calcular disimilitud como suma de diferencias cuadráticas
        mean_root_square_error = (np.sum((x1 - x2) ** 2 + (y1 - y2) ** 2) / n_samples) ** 0.5
        length_difference = self.abs_len_diff(other)

        dissimilarity_value = mean_root_square_error + c1 * length_difference


        if plot:
            plt.plot(x1, y1, 'o-', label='Arista 1', linewidth=2)
            plt.plot(x2, y2, 'o-', label='Arista 2', linewidth=2)
            plt.title(f'Dissimilarity: {dissimilarity_value:.2f} (tx={best_tx:.1f}, ty={best_ty:.1f})')
            plt.axis('equal')
            plt.legend()
            plt.show()

        # ploteamos un segmento de recta entre cada par de puntos para ver mejor la diferencia
        if plot:
            for i in range(n_samples):
                plt.plot([x1[i], x2[i]], [y1[i], y2[i]], 'r-', alpha=0.3)
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


        

class Backtrack_solver():
    def __init__(self, pieces = [Piece]):
        self.pieces = pieces

    def solve(self, plot_for_ipynb_t_1=False, plot_for_ipynb_t_2=False, plot_for_ipynb_t_3=False):


        def generate_spiral_index(n, m):
            """Genera una lista de indices de una matriz n x m en orden espiral."""
            result = []
            top, bottom, left, right = 0, n - 1, 0, m - 1

            while top <= bottom and left <= right:
                for j in range(left, right + 1):
                    result.append((top, j))
                top += 1

                for i in range(top, bottom + 1):
                    result.append((i, right))
                right -= 1

                if top <= bottom:
                    for j in range(right, left - 1, -1):
                        result.append((bottom, j))
                    bottom -= 1

                if left <= right:
                    for i in range(bottom, top - 1, -1):
                        result.append((i, left))
                    left += 1

            return result

        spiral_index = generate_spiral_index(5, 5)
        distancias = np.full((len(self.pieces)*4, len(self.pieces)*4), None)
        
        # Una solucion se compone por tuplas (pieza_index, rotacion) en orden espirar definido por spiral_index
        start_solution = []

        # Buscamos una esquina para la primera pieza
        for i, pieza in enumerate(self.pieces):
            edge_counter = 0

            if len(start_solution) > 0:
                break

            for rot in range(4):
                edge = pieza.edges[rot]
                if edge.kind == 'plano':
                    edge_counter += 1

                if edge_counter >= 2:
                    start_solution.append( (i, rot) )
                    break

        def r_solve(current_solution):

            def plot_t_1():
                """Este plot es el que recive el usuario como solucion y die que piezas va en cada posicion y con que rotacion. Para que el usuario lo pueda reconstruir manualmente."""
                
                clear_output(wait=True)

                fig, ax = plt.subplots(5, 5, figsize=(10, 10))

                for idx, (pieza_index, rotacion) in enumerate(current_solution):
                    plot_cords = spiral_index[idx]
                    pieza = self.pieces[pieza_index]
                    img = cv.cvtColor(cv.imread(pieza.back_img_path), cv.COLOR_BGR2RGB)

                    img_rotated = np.rot90(img, k=rotacion)

                    plt.sca(ax[plot_cords])
                    plt.title(f'Pieza {pieza_index}, rot {rotacion}', fontsize=6)
                    plt.imshow(img_rotated)
                    plt.axis('off')

                for idx in range(len(current_solution), 25):
                    plt.sca(ax[spiral_index[idx]])
                    plt.axis('off')

                plt.show()

            def plot_t_2():
                """Este plot muestra la reconstruccion desde la parte de atras de manera que las piezas estas lo mejor alineadas posible segun las esquinas detectadas."""

                path_masks = "fotos/segmented_scan_25/back_contours"

                canvas = np.zeros((2500, 2500, 3), dtype=np.uint8)
                
                clear_output(wait=True)


                piezas_colocadas = {}

                def rotate_corner(corner, k, img_shape):
                    """
                    Rota las coordenadas de una esquina k*90 grados antihorario (como np.rot90).
                    corner: (x, y) en coordenadas de imagen original
                    k: número de rotaciones de 90°
                    img_shape: (h, w) de la imagen ANTES de rotar
                    Retorna: (x, y) en coordenadas de imagen rotada
                    """
                    x, y = float(corner[0]), float(corner[1])
                    h, w = img_shape[0], img_shape[1]
                    
                    for _ in range(k % 4):
                        # Rotación 90° antihorario: (x, y) -> (y, w-1-x)
                        new_x = y
                        new_y = w - 1 - x
                        x, y = new_x, new_y
                        h, w = w, h
                    
                    return np.array([x, y])

                def get_edge_corners(pieza, edge_pos, rotacion, img_shape):
                    """
                    Obtiene las dos esquinas de una arista en coordenadas de imagen rotada.
                    edge_pos: posición de la arista después de rotar (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
                    rotacion: rotación aplicada a la imagen
                    Retorna: (c1, c2) donde c1 es la esquina "inicial" y c2 la "final" de la arista
                    """
                    edge_orig = (edge_pos + rotacion) % 4
                    
                    # Según generate_edges: c2 = corners[i], c1 = corners[(i+1)%4]
                    c1_orig = np.array(pieza.corners[(edge_orig + 1) % 4])
                    c2_orig = np.array(pieza.corners[edge_orig])
                    
                    c1_rot = rotate_corner(c1_orig, rotacion, img_shape)
                    c2_rot = rotate_corner(c2_orig, rotacion, img_shape)
                    
                    return c1_rot, c2_rot

                def get_all_corners_rotated(pieza, rotacion, img_shape):
                    """Obtiene las 4 esquinas rotadas de una pieza."""
                    corners_rot = []
                    for corner in pieza.corners:
                        corners_rot.append(rotate_corner(np.array(corner), rotacion, img_shape))
                    return corners_rot

                # Colocamos la primera pieza
                pieza_0_index, rotacion_0 = current_solution[0]
                pieza_0 = self.pieces[pieza_0_index]
                img_0 = cv.cvtColor(cv.imread(pieza_0.back_img_path), cv.COLOR_BGR2RGB)
                mask_0 = cv.imread(f"{path_masks}/{pieza_0_index:02d}.png", cv.IMREAD_GRAYSCALE)
                
                img_0_rotated = np.rot90(img_0, k=rotacion_0)
                mask_0_rotated = np.rot90(mask_0, k=rotacion_0)
                
                roi_0 = cv.bitwise_and(img_0_rotated, img_0_rotated, mask=mask_0_rotated)
                
                h0, w0 = roi_0.shape[:2]
                offset_x, offset_y = 600, 600
                
                canvas[offset_y:offset_y+h0, offset_x:offset_x+w0] = cv.add(
                    canvas[offset_y:offset_y+h0, offset_x:offset_x+w0], roi_0
                )
                
                corners_0_rot = get_all_corners_rotated(pieza_0, rotacion_0, img_0.shape)
                corners_0_canvas = [c + np.array([offset_x, offset_y]) for c in corners_0_rot]
                
                piezas_colocadas[0] = {
                    'offset': (offset_x, offset_y),
                    'img_shape_orig': img_0.shape,
                    'rotacion': rotacion_0,
                    'corners_canvas': corners_0_canvas
                }

                # Mapeo de dirección a aristas correspondientes
                direction_to_edges = {
                    (0, -1): (1, 3),   
                    (0, 1): (3, 1),    
                    (-1, 0): (2, 0),   
                    (1, 0): (0, 2),   
                }

                pos_to_spiral_idx = {pos: idx for idx, pos in enumerate(spiral_index)}

                for idx, (pieza_index, rotacion) in enumerate(current_solution[1:], start=1):
                    
                    pieza = self.pieces[pieza_index]
                    img = cv.cvtColor(cv.imread(pieza.back_img_path), cv.COLOR_BGR2RGB)
                    mask = cv.imread(f"{path_masks}/{pieza_index:02d}.png", cv.IMREAD_GRAYSCALE)
                    
                    img_rotated = np.rot90(img, k=rotacion)
                    mask_rotated = np.rot90(mask, k=rotacion)
                    
                    roi = cv.bitwise_and(img_rotated, img_rotated, mask=mask_rotated)
                    h_new, w_new = roi.shape[:2]

                    curr_i, curr_j = spiral_index[idx]
                    
                    # Buscar TODOS los vecinos ya colocados
                    vecinos_encontrados = []
                    for di, dj in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        vecino_pos = (curr_i + di, curr_j + dj)
                        if vecino_pos in pos_to_spiral_idx:
                            v_idx = pos_to_spiral_idx[vecino_pos]
                            if v_idx < idx:
                                vecinos_encontrados.append((v_idx, (di, dj)))
                    
                    if len(vecinos_encontrados) > 0:
                        # Calcular offset promediando las sugerencias de todos los vecinos
                        offsets_sugeridos = []
                        
                        for vecino_idx, direccion_desde_vecino in vecinos_encontrados:
                            arista_vecino_pos, arista_nueva_pos = direction_to_edges[direccion_desde_vecino]
                            
                            vecino_info = piezas_colocadas[vecino_idx]
                            vecino_pieza_idx, vecino_rot = current_solution[vecino_idx]
                            vecino_pieza = self.pieces[vecino_pieza_idx]
                            
                            # Esquinas de la arista del vecino
                            c1_vecino, c2_vecino = get_edge_corners(
                                vecino_pieza, arista_vecino_pos, vecino_rot, vecino_info['img_shape_orig']
                            )
                            
                            vx, vy = vecino_info['offset']
                            c1_vecino_canvas = c1_vecino + np.array([vx, vy])
                            c2_vecino_canvas = c2_vecino + np.array([vx, vy])
                            
                            # Esquinas de la arista de la nueva pieza
                            c1_nueva, c2_nueva = get_edge_corners(
                                pieza, arista_nueva_pos, rotacion, img.shape
                            )
                            
                            # Las aristas que conectan van en direcciones opuestas
                            # c1_vecino debe coincidir con c2_nueva y viceversa
                            mid_vecino = (c1_vecino_canvas + c2_vecino_canvas) / 2
                            mid_nueva = (c1_nueva + c2_nueva) / 2
                            
                            offset_sugerido = mid_vecino - mid_nueva
                            offsets_sugeridos.append(offset_sugerido)
                        
                        # Promediar todos los offsets sugeridos
                        offset_promedio = np.mean(offsets_sugeridos, axis=0)
                        offset_x_new = int(round(offset_promedio[0]))
                        offset_y_new = int(round(offset_promedio[1]))
                    else:
                        prev_info = piezas_colocadas[idx - 1]
                        offset_x_new = prev_info['offset'][0] + 150
                        offset_y_new = prev_info['offset'][1]

                    # Asegurar límites del canvas
                    offset_x_new = max(0, min(offset_x_new, canvas.shape[1] - w_new))
                    offset_y_new = max(0, min(offset_y_new, canvas.shape[0] - h_new))
                    
                    canvas[offset_y_new:offset_y_new+h_new, offset_x_new:offset_x_new+w_new] = cv.add(
                        canvas[offset_y_new:offset_y_new+h_new, offset_x_new:offset_x_new+w_new], roi
                    )
                    
                    corners_rot = get_all_corners_rotated(pieza, rotacion, img.shape)
                    corners_canvas = [c + np.array([offset_x_new, offset_y_new]) for c in corners_rot]
                    
                    piezas_colocadas[idx] = {
                        'offset': (offset_x_new, offset_y_new),
                        'img_shape_orig': img.shape,
                        'rotacion': rotacion,
                        'corners_canvas': corners_canvas
                    }
                
                plt.figure(figsize=(14, 14))
                plt.imshow(canvas)
                plt.axis('off')
                plt.show()

            def plot_t_3():
                """Este plot muestra la reconstruccion desde la parte de delante de manera que las piezas estas lo mejor alineadas posible segun las esquinas detectadas."""

                path_masks = "fotos/segmented_scan_25/front_contours"

                canvas = np.zeros((2500, 2500, 3), dtype=np.uint8)
                
                clear_output(wait=True)

                piezas_colocadas = {}

                def rotate_corner(corner, k, img_shape):

                    x, y = float(corner[0]), float(corner[1])
                    h, w = img_shape[0], img_shape[1]
                    
                    for _ in range(k % 4):
                        new_x = y
                        new_y = w - 1 - x
                        x, y = new_x, new_y
                        h, w = w, h
                    
                    return np.array([x, y])

                def get_edge_corners(pieza, edge_pos, rotacion, img_shape):

                    edge_orig = (edge_pos + rotacion) % 4
                    
                    c1_orig = np.array(pieza.corners[(edge_orig + 1) % 4])
                    c2_orig = np.array(pieza.corners[edge_orig])
                    
                    c1_rot = rotate_corner(c1_orig, rotacion, img_shape)
                    c2_rot = rotate_corner(c2_orig, rotacion, img_shape)
                    
                    return c1_rot, c2_rot

                def get_all_corners_rotated(pieza, rotacion, img_shape):

                    corners_rot = []
                    for corner in pieza.corners:
                        corners_rot.append(rotate_corner(np.array(corner), rotacion, img_shape))
                    return corners_rot


                pieza_0_index, rotacion_0 = current_solution[0]
                pieza_0 = self.pieces[pieza_0_index]
                img_0 = cv.cvtColor(cv.imread(pieza_0.front_img_path), cv.COLOR_BGR2RGB)
                mask_0 = cv.imread(f"{path_masks}/{pieza_0_index:02d}.png", cv.IMREAD_GRAYSCALE)
                
                img_0_flipped = cv.flip(img_0, 1)
                mask_0_flipped = cv.flip(mask_0, 1)
                
                img_0_transformed = np.rot90(img_0_flipped, k=rotacion_0)
                mask_0_transformed = np.rot90(mask_0_flipped, k=rotacion_0)
                
                roi_0 = cv.bitwise_and(img_0_transformed, img_0_transformed, mask=mask_0_transformed)
                
                h0, w0 = roi_0.shape[:2]
                offset_x, offset_y = 600, 600
                
                canvas[offset_y:offset_y+h0, offset_x:offset_x+w0] = cv.add(
                    canvas[offset_y:offset_y+h0, offset_x:offset_x+w0], roi_0
                )
                
                corners_0_rot = get_all_corners_rotated(pieza_0, rotacion_0, img_0.shape)
                corners_0_canvas = [c + np.array([offset_x, offset_y]) for c in corners_0_rot]
                
                piezas_colocadas[0] = {
                    'offset': (offset_x, offset_y),
                    'img_shape_orig': img_0.shape,
                    'rotacion': rotacion_0,
                    'corners_canvas': corners_0_canvas
                }

                # Mismo mapping que plot_t_2
                direction_to_edges = {
                    (0, -1): (1, 3),
                    (0, 1): (3, 1),
                    (-1, 0): (2, 0),
                    (1, 0): (0, 2),
                }

                pos_to_spiral_idx = {pos: idx for idx, pos in enumerate(spiral_index)}

                for idx, (pieza_index, rotacion) in enumerate(current_solution[1:], start=1):
                    
                    pieza = self.pieces[pieza_index]
                    img = cv.cvtColor(cv.imread(pieza.front_img_path), cv.COLOR_BGR2RGB)
                    mask = cv.imread(f"{path_masks}/{pieza_index:02d}.png", cv.IMREAD_GRAYSCALE)
                    
                    # Flip horizontal para alinear con BACK, luego rotar
                    img_flipped = cv.flip(img, 1)
                    mask_flipped = cv.flip(mask, 1)
                    
                    img_transformed = np.rot90(img_flipped, k=rotacion)
                    mask_transformed = np.rot90(mask_flipped, k=rotacion)
                    
                    roi = cv.bitwise_and(img_transformed, img_transformed, mask=mask_transformed)
                    h_new, w_new = roi.shape[:2]

                    curr_i, curr_j = spiral_index[idx]
                    
                    vecinos_encontrados = []
                    for di, dj in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        vecino_pos = (curr_i + di, curr_j + dj)
                        if vecino_pos in pos_to_spiral_idx:
                            v_idx = pos_to_spiral_idx[vecino_pos]
                            if v_idx < idx:
                                vecinos_encontrados.append((v_idx, (di, dj)))
                    
                    if len(vecinos_encontrados) > 0:
                        offsets_sugeridos = []
                        
                        for vecino_idx, direccion_desde_vecino in vecinos_encontrados:
                            arista_vecino_pos, arista_nueva_pos = direction_to_edges[direccion_desde_vecino]
                            
                            vecino_info = piezas_colocadas[vecino_idx]
                            vecino_pieza_idx, vecino_rot = current_solution[vecino_idx]
                            vecino_pieza = self.pieces[vecino_pieza_idx]
                            
                            c1_vecino, c2_vecino = get_edge_corners(
                                vecino_pieza, arista_vecino_pos, vecino_rot, vecino_info['img_shape_orig']
                            )
                            
                            vx, vy = vecino_info['offset']
                            c1_vecino_canvas = c1_vecino + np.array([vx, vy])
                            c2_vecino_canvas = c2_vecino + np.array([vx, vy])
                            
                            c1_nueva, c2_nueva = get_edge_corners(
                                pieza, arista_nueva_pos, rotacion, img.shape
                            )
                            
                            mid_vecino = (c1_vecino_canvas + c2_vecino_canvas) / 2
                            mid_nueva = (c1_nueva + c2_nueva) / 2
                            
                            offset_sugerido = mid_vecino - mid_nueva
                            offsets_sugeridos.append(offset_sugerido)
                        
                        offset_promedio = np.mean(offsets_sugeridos, axis=0)
                        offset_x_new = int(round(offset_promedio[0]))
                        offset_y_new = int(round(offset_promedio[1]))
                    else:
                        prev_info = piezas_colocadas[idx - 1]
                        offset_x_new = prev_info['offset'][0] + 150
                        offset_y_new = prev_info['offset'][1]

                    offset_x_new = max(0, min(offset_x_new, canvas.shape[1] - w_new))
                    offset_y_new = max(0, min(offset_y_new, canvas.shape[0] - h_new))
                    
                    canvas[offset_y_new:offset_y_new+h_new, offset_x_new:offset_x_new+w_new] = cv.add(
                        canvas[offset_y_new:offset_y_new+h_new, offset_x_new:offset_x_new+w_new], roi
                    )
                    
                    corners_rot = get_all_corners_rotated(pieza, rotacion, img.shape)
                    corners_canvas = [c + np.array([offset_x_new, offset_y_new]) for c in corners_rot]
                    
                    piezas_colocadas[idx] = {
                        'offset': (offset_x_new, offset_y_new),
                        'img_shape_orig': img.shape,
                        'rotacion': rotacion,
                        'corners_canvas': corners_canvas
                    }
                
                plt.figure(figsize=(14, 14))
                plt.imshow(canvas)
                plt.axis('off')
                plt.show()


            if plot_for_ipynb_t_1:
                plot_t_1()

            if plot_for_ipynb_t_2:
                plot_t_2()

            if plot_for_ipynb_t_3:
                plot_t_3()


            c1 = 0.25
            n_samples = 30
            greedy = True
            edge_threshold = 9
            interior_threshold = 16

            #c1 = 2
            #n_samples = 20
            #greedy = False
            #edge_threshold = 9
            #interior_threshold = 16
            
            # Tamao de borde 5x5
            if len(current_solution) < 16:

                next_pos = spiral_index[len(current_solution)]
                next_i, next_j = next_pos

                dif = (next_i - spiral_index[len(current_solution)-1][0], next_j - spiral_index[len(current_solution)-1][1])
                mapping_dif_to_edge_id = {
                    (0, 1): (1, 3),   # ir derecha
                    (1, 0): (2, 0),   # ir abajo: arista abajo(2) de anterior, arriba(0) de nueva
                    (0, -1): (3, 1),  # ir izquierda
                    (-1, 0): (0, 2)   # ir arriba: arista arriba(0) de anterior, abajo(2) de nueva
                }

                required_edges = mapping_dif_to_edge_id[dif]
                first_edge_id, second_edge_id = required_edges

                pares_pieza_arista_candidatos = []

                for pieza_index, pieza in enumerate(self.pieces):

                    # Comprobamos que la pieza este disponible para usar (No haya sido usada en la solucion actual)
                    if pieza_index in [s[0] for s in current_solution]:
                        continue  # ya usada

                    # comprobamos que la pieza tenga al menos un borde (Estamos resolviendo primero el exterior)
                    is_edge = False
                    for rot in range(4):
                        edge = pieza.edges[rot]
                        if edge.kind == 'plano':
                            is_edge = True
                    if not is_edge:
                        continue

                    for rot in range(4):
                        edge = pieza.edges[rot]
                        if edge.kind != 'plano':
                            pares_pieza_arista_candidatos.append( (pieza_index, rot) )

                
                # Funcion que usaremos como key para ordenar los candidatos por disimilaridad
                def dissimilarity_key(id_pieza, edge_id):
                    return self.pieces[id_pieza].edges[edge_id].dissimilarity(

                        self.pieces[current_solution[-1][0]].edges[
                            (current_solution[-1][1] + first_edge_id) % 4
                        ], c1=c1, n_samples=n_samples, plot=False
                    )
                
                pares_pieza_arista_candidatos = sorted(
                    pares_pieza_arista_candidatos,
                    key=lambda x: dissimilarity_key(x[0], x[1])
                )

                # Caso base none si el mejor candidato supera un umbral de disimilaridad
                
                if self.pieces[current_solution[-1][0]].edges[
                    (current_solution[-1][1] + first_edge_id) % 4
                ].dissimilarity(
                    self.pieces[pares_pieza_arista_candidatos[0][0]].edges[
                        pares_pieza_arista_candidatos[0][1]
                    ], c1=c1, n_samples=n_samples, plot=False
                ) > edge_threshold:
                    if not greedy:
                        return None

                for pieza_index, edge_id in pares_pieza_arista_candidatos:
                    # probamos a añadir la pieza a la solucion
                    new_solution = current_solution + [ (pieza_index, (edge_id - second_edge_id) % 4) ]

                    # llamamos recursivamente
                    result = r_solve(new_solution)
                    if result is not None:
                        return result
            
            elif len(current_solution) < 25:
                
                next_pos = spiral_index[len(current_solution)]
                next_i, next_j = next_pos

                # Crear diccionario de posición a índice en espiral para buscar vecinos
                pos_to_spiral_idx = {pos: idx for idx, pos in enumerate(spiral_index)}

                # Mapping de dirección relativa a (arista_del_candidato, arista_del_vecino)
                # Si el vecino está arriba del candidato, el candidato conecta con su arista 0 (arriba)
                # y el vecino conecta con su arista 2 (abajo)
                direction_to_edges = {
                    (-1, 0): (0, 2),  # vecino arriba: candidato usa arista 0, vecino usa arista 2
                    (1, 0): (2, 0),   # vecino abajo: candidato usa arista 2, vecino usa arista 0
                    (0, -1): (3, 1),  # vecino izquierda: candidato usa arista 3, vecino usa arista 1
                    (0, 1): (1, 3),   # vecino derecha: candidato usa arista 1, vecino usa arista 3
                }

                # Encontrar todos los vecinos adyacentes que ya están en la solución
                vecinos_resueltos = []  # Lista de (spiral_idx, direccion)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    vecino_pos = (next_i + di, next_j + dj)
                    if vecino_pos in pos_to_spiral_idx:
                        vecino_spiral_idx = pos_to_spiral_idx[vecino_pos]
                        if vecino_spiral_idx < len(current_solution):
                            # Este vecino ya está resuelto
                            vecinos_resueltos.append((vecino_spiral_idx, (di, dj)))

                pares_pieza_arista_candidatos = []

                for pieza_index, pieza in enumerate(self.pieces):

                    # Comprobamos que la pieza este disponible para usar
                    if pieza_index in [s[0] for s in current_solution]:
                        continue  # ya usada

                    # Para piezas interiores: descartamos las que tienen borde plano
                    has_flat_edge = False
                    for rot in range(4):
                        edge = pieza.edges[rot]
                        if edge.kind == 'plano':
                            has_flat_edge = True
                            break
                    if has_flat_edge:
                        continue

                    # Añadimos todas las rotaciones posibles (ninguna arista es plana en piezas interiores)
                    for rot in range(4):
                        pares_pieza_arista_candidatos.append((pieza_index, rot))

                
                # Funcion que calcula la disimilaridad total con TODOS los vecinos resueltos
                def dissimilarity_key_interior(id_pieza, rotacion_candidato):
                    total_dissimilarity = 0
                    
                    for vecino_spiral_idx, direccion in vecinos_resueltos:
                        # Obtener la pieza vecina y su rotación
                        vecino_pieza_idx, vecino_rotacion = current_solution[vecino_spiral_idx]
                        
                        # Obtener qué aristas conectan según la dirección
                        arista_pos_candidato, arista_pos_vecino = direction_to_edges[direccion]
                        
                        # Calcular la arista original del candidato que está en esa posición
                        # Si rotacion_candidato = R, la arista en posición P es la arista original (P + R) % 4
                        arista_candidato = (arista_pos_candidato + rotacion_candidato) % 4
                        
                        # Calcular la arista original del vecino que está en esa posición
                        arista_vecino = (arista_pos_vecino + vecino_rotacion) % 4
                        
                        # Calcular disimilaridad entre estas aristas
                        dissim = self.pieces[id_pieza].edges[arista_candidato].dissimilarity(
                            self.pieces[vecino_pieza_idx].edges[arista_vecino],
                            c1=c1, n_samples=n_samples, plot=False
                        )
                        total_dissimilarity += dissim
                    
                    return total_dissimilarity
                
                pares_pieza_arista_candidatos = sorted(
                    pares_pieza_arista_candidatos,
                    key=lambda x: dissimilarity_key_interior(x[0], x[1])
                )

                # Si no hay candidatos válidos, retroceder
                if len(pares_pieza_arista_candidatos) == 0:
                    return None

                # Caso base: si el mejor candidato supera un umbral de disimilaridad
                best_dissimilarity = dissimilarity_key_interior(
                    pares_pieza_arista_candidatos[0][0],
                    pares_pieza_arista_candidatos[0][1]
                )
                
                # Umbral ajustado: consideramos que hay más vecinos, así que el umbral total es mayor
                umbral_por_vecino = interior_threshold
                if best_dissimilarity > umbral_por_vecino * len(vecinos_resueltos):
                    if not greedy:  # Backtrack si no hay buen candidato
                        return None

                for pieza_index, rotacion in pares_pieza_arista_candidatos:
                    # Añadimos la pieza con su rotación directamente
                    new_solution = current_solution + [(pieza_index, rotacion)]

                    # llamamos recursivamente
                    result = r_solve(new_solution)
                    if result is not None:
                        return result
                
                return None  # Ningún candidato funcionó, backtrack

            # Solución completa (25 piezas)
            if len(current_solution) == 25:
                return current_solution
            
            return None

            
                    


            
        return r_solve(start_solution)






def main():
    # Cargamos las piezas
    lista_piezas = [Piece(i)for i in range(25)]
    for i in range(5):
        lista_piezas[i].edges[0].kind = "plano"
        lista_piezas[i+20].edges[2].kind = "plano"
        if i == 0:
            for j in range(5):
                lista_piezas[i+5*j].edges[1].kind = "plano"
        if i == 4:
            for j in range(5):
                lista_piezas[i+5*j].edges[3].kind = "plano"

    solver = Backtrack_solver(lista_piezas)
    solution = solver.solve()

    print("Solución encontrada:")
    print(solution)

if __name__ == "__main__":
    main()