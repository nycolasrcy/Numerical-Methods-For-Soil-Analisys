import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
a = 110
length = 50
time = 1000
nodes = 300

dx = length / nodes
dt = 0.5 * dx**2 / a
t_nodes = int(time / dt)

# Inicialização
u = np.zeros(nodes) + 20
u[0] = 100
u[-1] = 100

# Configuração do gráfico
fig, axis = plt.subplots()
pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)
axis.set_ylim([-2, 3])

# Tempo de execução
for counter in np.arange(0, time, dt):
    
    # Cálculo da evolução usando vetorização
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + a * dt * (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
    u = u_new

    # Atualização do gráfico
    pcm.set_array(u[None, :].flatten())
    axis.set_title(f"Distribution at t: {counter:.3f} [s].")
    plt.pause(0.01)  # Ajuste ou remova a pausa se não for necessário


plt.show()
