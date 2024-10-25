from camclay import *
from tensor import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
analisys = ModifiedCamClay()

# tensors

stress_tensor_trial = np.zeros((3,3), float)
stress_tensor_converged = np.zeros((3,3), float)
dstrain_tensor_total = np.zeros((3,3), float)
dstrain_tensor_plastic = np.zeros((3,3), float)

# initial stress state and inital state variables

pini = 100

stress_tensor_converged[0,0] = pini
stress_tensor_converged[1,1] = pini
stress_tensor_converged[2,2] = pini

analisys.update_state_variables(0, stress_tensor_converged, analisys.check_for_plasticity(0,stress_tensor_converged))

depsilon = 8e-5 #8e-5
dstrain_tensor_total[0,0] = depsilon
dstrain_tensor_total += - (depsilon / 3) * np.eye(3)

elastoplastic_modulus = analisys.build_elastoplastic_operator(0, stress_tensor_converged, stress_tensor_trial, -1, 0)
print(elastoplastic_modulus)

for index in range(analisys.maxiter):

    stress_tensor_trial = stress_tensor_converged + np.einsum('ijkl, kl -> ij', elastoplastic_modulus, dstrain_tensor_total)

    F = analisys.check_for_plasticity(index, stress_tensor_trial)
    #print(stress_tensor_trial)
    
    if F < 0: # elastic domain - update state variables according to elastic behavior

        stress_tensor_converged = stress_tensor_trial
        analisys.update_state_variables(index, stress_tensor_converged, F)
        elastoplastic_modulus = analisys.build_elastoplastic_operator(index, stress_tensor_converged, stress_tensor_trial, F, 0)

        analisys.data[index,0] = volumetric_scalar(stress_tensor_converged)
        analisys.data[index,1] = deviatoric_scalar(stress_tensor_converged)

    else: # plastic domain - update state variables according to elastoplastic behavior
        
        try:

            dphi, p, q = analisys.compute_plastic_mulitplier(index, stress_tensor_trial)
        
        except Exception as error:
            
            print(error)
            break

        n = deviatoric_tensor(stress_tensor_trial) / np.linalg.norm(deviatoric_tensor(stress_tensor_trial))
        derivative = (1 / 3) * (2 * p - analisys.pc) * np.eye(3) + np.sqrt(3 / 2) * (2 * q / analisys.M**2) * n
        dstrain_tensor_plastic = dphi * derivative

        stress_tensor_converged = stress_tensor_trial - np.tensordot(elastoplastic_modulus, dstrain_tensor_plastic, axes = [(2,3), (0,1)])
       
        analisys.update_state_variables(index, stress_tensor_converged, F)

        elastoplastic_modulus = analisys.build_elastoplastic_operator(index, stress_tensor_converged, stress_tensor_trial, F, dphi)

        analisys.data[index, 0] = volumetric_scalar(stress_tensor_converged)
        analisys.data[index, 1] = deviatoric_scalar(stress_tensor_converged)

df = pd.DataFrame(analisys.data)
df.to_excel('log.xlsx', index = False)


def plot_stress_path_CSL_yield_surface(data, M, pc):
    # Criando a figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extraindo p (pressão média) e q (desviador de tensões) do array de dados
    p_values = data[:, 0]  # coluna 0 tem p
    q_values = data[:, 1]  # coluna 1 tem q

    # Plotando o caminho de tensões (stress path)
    ax.plot(p_values, q_values, label='Stress Path', color='b', marker='o')

    # Plotando a CSL (Critical State Line) como uma linha reta: q = M * p
    p_CSL = np.linspace(0, max(p_values) * 1.1, 100)
    q_CSL = M * p_CSL
    ax.plot(p_CSL, q_CSL, label='CSL: q = M * p', color='g', linestyle='--')

    # Plotando a superfície de escoamento inicial com a fórmula corrigida
    p_yield = np.linspace(0, max(p_values) * 1.1, 100)
    q_yield = M * np.sqrt(-(p_yield * (p_yield - pc)))
    
    # Evitar valores complexos: vamos definir q_yield = 0 para p > pc
    q_yield[p_yield > pc] = 0  
    
    ax.plot(p_yield, q_yield, label='Initial Yield Surface', color='r', linestyle='-')

    # Configurações do gráfico
    ax.set_xlabel('p (Mean Stress)')
    ax.set_ylabel('q (Deviatoric Stress)')
    ax.set_title('Stress Path, CSL, and Initial Yield Surface')
    ax.legend()
    ax.grid(True)

    plt.show()

plot_stress_path_CSL_yield_surface(analisys.data, analisys.M, analisys.pcini)
