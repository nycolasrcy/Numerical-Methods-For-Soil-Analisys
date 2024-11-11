import numpy as np
import pandas as pd
from ..tensor import *
from ..material import *

class ModifiedCamClay:

    def __init__(self, material):

        self.k = material.k # swelling effective pressure line slope
        self.l = material.l # normal effective pressure line slop
        self.M = material.M # critical state line slope
        self.v = material.v # poisson ratio
        self.e = material.e # void ratio
        self.pc = material.pc # pre-consolidation effective volumetric invariant

        self.K: float # effective bulk modulus
        self.E: float # effective elastic modulus
        self.G: float # effective shear modulus
        self.L: float # lamé constant
        self.O: float # state variable dependent on void ratio

        self.stress_tensor_converged: np.ndarray = np.zeros((3, 3), float)
        self.elastoplastic_operator: np.ndarray = np.zeros((3, 3, 3, 3), float)

        # set initial state of stresses and initial state variables
        self.stress_tensor_converged[0,0] = 100
        self.stress_tensor_converged[0,0] = 100
        self.stress_tensor_converged[0,0] = 100
    
    def update_state_variable(self, index: int, stress_tensor_converged: np.ndarray):

        p = volumetric_invariant(stress_tensor_converged)

        self.K = ((1 + self.e) * p) / self.k
        self.E = 3 * self.K * (1 - 2 * self.v)
        self.G = self.E / (2 * (1 + 2 * self.v))
        self.L = self.E * self.v / ((1 + self.v) * (1 - 2 * self.v))
        self.O = (1 + self.e) / (self.l - self.k)

    def check_for_plasticity(self, index: int, stress_tensor: np.ndarray) -> float:
        
        p = volumetric_invariant(stress_tensor)
        q = deviatoric_invariant(stress_tensor)

        F = (q / self.M)**2 + p * (p - self.pc)
        
        return F
    
    def compute_plastic_multiplier(self, index: int, stress_tensor_trial: np.ndarray) -> float:

        # iteration scheme parameters
        tolerance = 1e-5
        maxiteration = 50
        
        ptrial = volumetric_invariant(stress_tensor_trial)
        qtrial = deviatoric_invariant(stress_tensor_trial)
        pcn = self.pc 

        # initial guesses
        dphi = 0
        pc = self.pc

        for i in range(maxiteration): # local iteration scheme to compute dphi

            for j in range(maxiteration): # sublocal iteration scheme to compute pc

                R = pcn * np.exp(self.O * dphi * ((2 * ptrial - pc) / (1 + 2 * self.K * dphi))) - pc
                Rprime = pcn * np.exp(self.O * dphi * ((2 * ptrial - pc) / (1 + 2 * self.K * dphi))) * ((- self.O * dphi) / (1 + 2 * self.K * dphi)) - 1 
                
                if abs(R) <= tolerance:

                    p = (ptrial + dphi * self.K * pc) / (1 + 2 * self.K * dphi)
                    q = (qtrial) / (1 + (6 * self.G * dphi) / (self.M**2))

                    A = - (self.K * (2 * p - pc)**2) / (1 + (2 * self.K + self.O * pc) * dphi)
                    B = - (2 * q**2) / ((self.M**2) * (dphi + ((self.M**2) / (6 * self.G))))
                    C = - (p * self.O * pc) * ((2 * p - pc) / (1 + (2 * self.K + self.O * pc) * dphi))

                    F = (q / self.M)**2 + p * (p - self.pc)
                    Fprime = A + B + C

                    break

                pc = pc - (R / Rprime)

            else:

                raise RuntimeError(f"Sublocal Newton-Raphson's scheme did not converge! Global iteration counter: {index}.")
            
            if abs(F) <= tolerance:

                self.pc = pc # update the hardening parameter
                return dphi
            
            dphi = dphi - (F / Fprime)
        
        else:

            raise RuntimeError(f"Local Newton-Raphson's scheme did not converge! Global iteration counter: {index}.")            
     
    def build_elastoplastic_operator(self, index: int, stress_tensor_converged: np.ndarray, stress_tensor_trial: np.ndarray, dphi: float) -> np.ndarray:
 
        delta = np.eye(3)
        identity = np.einsum('ij,kl -> ijkl', delta, delta)
        isotropic = 0.5 * (np.einsum('ik, jl -> ijkl', delta, delta) + np.einsum('il, jk -> ijkl', delta, delta))

        deviatoric_norm_converged = np.linalg.norm(deviatoric_tensor(stress_tensor_converged), 'fro')
        deviatoric_norm_trial = np.linalg.norm(deviatoric_tensor(stress_tensor_trial), 'fro')
        
        xi = deviatoric_norm_converged / deviatoric_norm_trial
        n = deviatoric_tensor(stress_tensor_trial) / np.linalg.norm(deviatoric_tensor(stress_tensor_trial))

        p = volumetric_invariant(stress_tensor_converged)
        q = deviatoric_invariant(stress_tensor_converged)

        a0 = 1 + 2 * self.K * dphi + self.pc * self.O * dphi
        a1 = (1 + self.pc * self.O * dphi) / a0
        a2 = - (2 * p - self.pc) / a0
        a3 = 2 * self.pc * self.O * dphi / a0
        a4 = self.O * (self.pc / self.K) * (2 * p - self.pc) / a0
        a5 = np.sqrt(3 / 2) / (1 + 6 * self.G * (dphi / self.M**2))
        a6 = - ( 3 * q / self.M**2) / (1 + 6 * self.G * (dphi / self.M**2))

        b0 = - 2 * self.G * (2 * q / self.M**2) * a6 - self.K * ((2 * a2 - a4) * p - a2 * self.pc)
        b1 = - self.K * ((a3 - 2 * a1) * p + a1 * self.pc) / b0
        b2 = 2 * self.G * (2 * q / self.M**2) * (a5 / b0)

        elastoplastic_operator = 2 * self.G * xi * isotropic \
        + (self.K * (a1 + a2 * b1) - (1 / 3) * (2 * self.G) * xi) * identity \
        + self.K * (a2 * b2) * np.einsum('ij, kl -> ijkl', delta, n) \
        + 2 * self.G * np.sqrt(2 / 3) * (a6 * b1) * np.einsum('ij, kl -> ijkl', n, delta) \
        + 2 * self.G * (np.sqrt(2 / 3) * (a5 + a6 * b2) - xi) * np.einsum('ij, kl -> ijkl', n, n)

        return elastoplastic_operator
    
    def return_mapping_algorithm(self, index: int, dstrain_tensor_total: np.ndarray):

        # set inital condition FIXME: This is not optimal, this parte of code should be in the constructor __init__
        # the way this is implemented is that we are using the model function to modified the converged_stress_tensor
        # and the elastoplastic_operator of the __init__, that's why we have so many autorefer in the code below
        if index == 0:

            self.update_state_variable(index, self.stress_tensor_converged)
            
            self.elastoplastic_operator = self.L * np.einsum('ij, kl -> ijkl', np.eye(3), np.eye(3)) \
            + 2 * self.G * (0.5 * (np.einsum('ik, jl -> ijkl', np.eye(3), np.eye(3)) + np.einsum('il, jk -> ijkl', np.eye(3), np.eye(3))))

        # return mapping algorithm
        stress_tensor_trial = self.stress_tensor_converged + np.einsum('ijkl, kl -> ij', self.elastoplastic_modulus, dstrain_tensor_plastic)
        
        F = self.check_for_plasticity(index, stress_tensor_trial)

        if F < 0: # elastic region

            self.stress_tensor_converged = stress_tensor_trial
            self.update_state_variable(index, self.stress_tensor_converged)
            self.elastoplastic_operator = self.L * np.einsum('ij, kl -> ijkl', np.eye(3), np.eye(3)) \
            + 2 * self.G * (0.5 * (np.einsum('ik, jl -> ijkl', np.eye(3), np.eye(3)) + np.einsum('il, jk -> ijkl', np.eye(3), np.eye(3))))

        else: # plastic region

            p = volumetric_invariant(self.stress_tensor_converged)
            q = deviatoric_invariant(self.stress_tensor_converged)
            pc = self.pc

            try:

                dphi = self.compute_plastic_multiplier(index, stress_tensor_trial)

            except Exception as error:

                return error # when this occour: (I) soil failed -> stop FEM iteration (II) big shit had happened on the code so the NR scheme couldn't converge -> check what's going on
            
            n = deviatoric_tensor(stress_tensor_trial) / np.linalg.norm(deviatoric_tensor(stress_tensor_trial), 'fro')
            derivative = (1 / 3) * (2 * p - pc) * np.eye(3) + np.sqrt(3 / 2) * (2 * q / self.M**2) * n
            dstrain_tensor_plastic = dphi * derivative

            self.stress_tensor_converged = stress_tensor_trial - np.einsum('ijkl,kl -> ij', self.elastoplastic_operator, dstrain_tensor_plastic)
            self.update_state_variable(index, self.stress_tensor_converged)
            self.elastoplastic_operator = self.build_elastoplastic_operator(index, self.stress_tensor_converged, stress_tensor_trial, dphi)

def trixial_test():
    
    '''
    Testando o comportamento do modelo e de suas funções em uma simulação de ensaio
    triaxial não-drenado sobre um ponto material
    '''

    maxiter = 1000
    data = np.zeros((maxiter, 2), float)

    undrained = ModifiedCamClay(argila)

    stress_tensor_converged = np.zeros((3,3), float)
    dstrain_tensor_total = np.zeros((3,3), float)
    elastoplastic_operator = np.zeros((3,3,3,3), float)

    stress_tensor_converged[0,0] = 25
    stress_tensor_converged[1,1] = 25
    stress_tensor_converged[2,2] = 25

    undrained.update_state_variable(0, stress_tensor_converged)

    depsilon = 8e-5 #8e-5
    dstrain_tensor_total[0,0] = depsilon
    dstrain_tensor_total += - (depsilon / 3) * np.eye(3)

    delta = np.eye(3)
    identity = np.einsum('ij, kl -> ijkl', delta, delta)
    isotropic = 0.5 * (np.einsum('ik, jl -> ijkl', delta, delta) + np.einsum('il, jk -> ijkl', delta, delta))
    elastoplastic_operator = undrained.L * identity + 2 * undrained.G * isotropic

    for index in range(maxiter):

        stress_tensor_trial = stress_tensor_converged + np.einsum('ijkl, kl -> ij', elastoplastic_operator, dstrain_tensor_total)

        F = undrained.check_for_plasticity(index, stress_tensor_trial)

        if F < 0:

            stress_tensor_converged = stress_tensor_trial
            undrained.update_state_variable(index, stress_tensor_converged)
            elastoplastic_operator = undrained.L * identity + 2 * undrained.G * isotropic
            
            data[index, 0] = volumetric_invariant(stress_tensor_converged)
            data[index, 1] = deviatoric_invariant(stress_tensor_converged)
        else:

            p = volumetric_invariant(stress_tensor_converged)
            q = deviatoric_invariant(stress_tensor_converged)
            pc = undrained.pc

            try:

                dphi = undrained.compute_plastic_multiplier(index, stress_tensor_trial)
                print(index, dphi)

            except Exception as error:
                
                print(error)
                break
            
            n = deviatoric_tensor(stress_tensor_trial) / np.linalg.norm(deviatoric_tensor(stress_tensor_trial), 'fro')
            derivative = (1 / 3) * (2 * p - pc) * np.eye(3) + np.sqrt(3 / 2) * (2 * q / undrained.M**2) * n
            dstrain_tensor_plastic = dphi * derivative

            stress_tensor_converged = stress_tensor_trial - np.einsum('ijkl,kl -> ij', elastoplastic_operator, dstrain_tensor_plastic)
            undrained.update_state_variable(index, stress_tensor_converged)
            elastoplastic_operator = undrained.build_elastoplastic_operator(index, stress_tensor_converged, stress_tensor_trial, dphi)

            data[index, 0] = volumetric_invariant(stress_tensor_converged)
            data[index, 1] = deviatoric_invariant(stress_tensor_converged)

    df = pd.DataFrame(data)
    df.to_excel('log.xlsx', index = False)
