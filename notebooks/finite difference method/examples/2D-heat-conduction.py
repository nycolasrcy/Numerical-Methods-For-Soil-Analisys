import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
a = 110
length = 50
time = 100
nodes = 100

dx = length / nodes
dy = length / nodes

dt = min(dx**2 /(4*a), dy**2 / (4*a))
t_nodes = int(time / dt)

# Inicialização
u = np.zeros((nodes, nodes)) + 20

# condições de contorno
u[0, :] = np.linspace(0, 100, nodes)
u[-1, :] = np.linspace(0, 100, nodes)
u[:, 0] = np.linspace(0, 100, nodes)
u[:, -1] = np.linspace(0, 100, nodes)

# Configuração do gráfico
fig, axis = plt.subplots()
pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)

# Tempo de execução
for counter in np.arange(0, time, dt):
    
    # Cálculo da evolução usando vetorização
    u_new = u.copy()
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + a * dt * (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    u = u_new

    # Atualização do gráfico
    pcm.set_array(u.flatten())
    axis.set_title(f"Distribution at t: {counter:.3f} [s].")
    plt.pause(0.0001)  

plt.show()
