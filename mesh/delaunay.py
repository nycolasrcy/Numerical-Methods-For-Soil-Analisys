import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def delaunay_triangulation(points):
    """
    Função que realiza a triangulação de Delaunay em uma nuvem de pontos.
    
    Parâmetros:
    - points (ndarray): Nuvem de pontos, onde cada linha representa um ponto (x, y).
    
    Retorna:
    - tri (Delaunay): Objeto da triangulação de Delaunay gerado pela biblioteca SciPy.
    """
    # Realiza a triangulação de Delaunay
    tri = Delaunay(points)
    return tri

def plot_triangulation(points, tri):
    """
    Função que plota a triangulação de Delaunay para uma nuvem de pontos.
    
    Parâmetros:
    - points (ndarray): Nuvem de pontos, onde cada linha representa um ponto (x, y).
    - tri (Delaunay): Objeto da triangulação de Delaunay.
    """
    plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
    plt.plot(points[:, 0], points[:, 1], 'o', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triangulação de Delaunay')
    plt.show()

'''
# Exemplo de uso:
# Definindo uma nuvem de pontos (em um array numpy) que cobre uma área de geometria desejada
points = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1],  # Ponto de contorno
    [0.5, 0.5], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]  # Pontos internos
])

# Realiza a triangulação
tri = delaunay_triangulation(points)

# Plota a triangulação
plot_triangulation(points, tri)
'''

def create_uniform_grid(width, height, num_x_points, num_y_points):
    """
    Cria uma grade uniforme de pontos dentro de um retângulo.
    
    Parâmetros:
    - width (float): Largura do retângulo.
    - height (float): Altura do retângulo.
    - num_x_points (int): Número de pontos na direção x.
    - num_y_points (int): Número de pontos na direção y.
    
    Retorna:
    - points (ndarray): Array com os pontos da grade uniforme.
    """
    x = np.linspace(0, width, num_x_points)
    y = np.linspace(0, height, num_y_points)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    return points

# Parâmetros da geometria
width = 10  # Largura do retângulo
height = 5  # Altura do retângulo
num_x_points = 20  # Número de pontos na direção x
num_y_points = 10  # Número de pontos na direção y

# Gera os pontos para uma malha uniforme no retângulo
points = create_uniform_grid(width, height, num_x_points, num_y_points)

# Realiza a triangulação
tri = delaunay_triangulation(points)

# Plota a triangulação
plot_triangulation(points, tri)
