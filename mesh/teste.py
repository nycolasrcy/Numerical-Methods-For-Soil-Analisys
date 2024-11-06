import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import matplotlib.patches as patches

def delaunay_triangulation(points):
    """
    Realiza uma triangulação de Delaunay em uma nuvem de pontos.
    
    Parâmetros:
    - points (ndarray): Nuvem de pontos, onde cada linha representa um ponto (x, y).
    
    Retorna:
    - tri (Delaunay): Objeto da triangulação de Delaunay gerado pela biblioteca SciPy.
    """
    return Delaunay(points)

def generate_irregular_quadrilaterals(tri):
    """
    Agrupa triângulos adjacentes da triangulação de Delaunay para formar elementos quadrilaterais.
    
    Parâmetros:
    - tri (Delaunay): Objeto da triangulação de Delaunay.
    
    Retorna:
    - quadrilaterals (list of list): Lista de quadriláteros, cada um representado pelos índices dos nós.
    """
    triangles = tri.simplices
    neighbors = tri.neighbors
    quadrilaterals = []
    used_triangles = set()

    for i, triangle in enumerate(triangles):
        if i in used_triangles:
            continue
        
        # Busca um triângulo vizinho que compartilhe uma aresta
        for neighbor_index in neighbors[i]:
            if neighbor_index != -1 and neighbor_index not in used_triangles:
                # Verifica se os triângulos compartilham uma aresta (dois nós em comum)
                shared_nodes = set(triangle).intersection(triangles[neighbor_index])
                if len(shared_nodes) == 2:
                    # Cria um quadrilátero combinando os nós dos dois triângulos
                    quad_nodes = list(set(triangle).union(triangles[neighbor_index]))
                    quadrilaterals.append(quad_nodes)
                    used_triangles.add(i)
                    used_triangles.add(neighbor_index)
                    break

    return quadrilaterals

def plot_irregular_quadrilateral_mesh(points, quadrilaterals):
    """
    Plota a malha quadrilateral irregular.
    
    Parâmetros:
    - points (ndarray): Array de coordenadas dos nós (pontos).
    - quadrilaterals (list of list): Lista de elementos quadrilaterais, cada um representado pelos índices dos nós.
    """
    fig, ax = plt.subplots()
    for quad in quadrilaterals:
        # Obtém as coordenadas dos nós de cada quadrilátero
        polygon = points[quad, :]
        # Cria um polígono para cada quadrilátero e plota
        patch = patches.Polygon(polygon, closed=True, edgecolor='blue', fill=False)
        ax.add_patch(patch)

    # Plota os nós
    ax.plot(points[:, 0], points[:, 1], 'o', color='red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Malha Quadrilateral Irregular")
    ax.set_aspect('equal')
    plt.show()

# Exemplo de uso:
# Define uma nuvem de pontos semi-aleatória dentro de um retângulo
points = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
    [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
    [0, 3], [1, 3], [2, 3], [3, 3], [4, 3]
])

# Realiza a triangulação de Delaunay
tri = delaunay_triangulation(points)

# Gera os elementos quadrilaterais irregulares a partir dos triângulos
quadrilaterals = generate_irregular_quadrilaterals(tri)

# Plota a malha quadrilateral irregular
plot_irregular_quadrilateral_mesh(points, quadrilaterals)
