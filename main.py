from geometry import Geometry, Mesh
from material import Material
from elements import QuadElement
from boundary_conditions import BoundaryConditions
from assembler import assemble
from solver import Solver
from post_processing import post_process

# 1. Defina a geometria e a malha
geometry = Geometry(...)
mesh = Mesh(geometry)

# 2. Defina o material
material = Material(E=210e9, nu=0.3)

# 3. Configure os elementos e calcule suas matrizes de rigidez
elements = [QuadElement(nodes, material) for nodes in mesh.elements]

# 4. Aplique as condições de contorno
bc = BoundaryConditions(...)
bc.apply(...)

# 5. Monte a matriz global e o vetor de forças
K, F = assemble(elements, mesh, bc)

# 6. Resolva o sistema
solver = Solver(K, F)
displacements = solver.solve()

# 7. Pós-processamento
post_process(displacements, mesh)
