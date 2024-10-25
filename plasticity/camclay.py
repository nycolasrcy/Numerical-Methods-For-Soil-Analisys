import numpy as np
from tensor import *

class ModifiedCamClay:

    def __init__(self):

        self.k: float = 0.10 # swelling pressure line slope
        self.l: float = 0.01 # normal pressure line slope
        self.M: float = 1.20 # critical state line slop
        self.v: float = 0.35 # poisson modulus

        self.e: float = 1.00 # void ratio
        self.pc: float = 150 # pre-consolidation pressure [kPa]
        self.pcini: float = self.pc

        self.K: float # bulk modulus
        self.E: float # elastic modulus
        self.G: float # shear modulus
        self.L: float # lamé constant
        self.O: float # state variable dependent on void ratio

        self.maxiter: int = 10000 # max interation for the simulation
        self.data: np.ndarray = np.zeros((self.maxiter, 7), float) # data structure that storages information of each time-step

        self.strain_tensor_total = np.zeros((3,3), float)
        self.strain_tensor_elastic = np.zeros((3,3), float)
        self.strain_tensor_plastic = np.zeros((3,3), float)

    def update_state_variables(self, index: int, stress_tensor_converged: np.ndarray, yielding_function_value: float):

        p =  volumetric_scalar(stress_tensor_converged)
        pn = self.data[index - 1, 0]

        '''if index == 0: # first iteration, so no need to update the void ratio

            self.e = self.e

        else:
        
            if yielding_function_value < 0: # update void ratio explicitly according to elastic behavior

                self.e -= self.k * np.log(pn / p)

            else: # update void ratio explicitly according to inelastic behavior

                self.e -= self.l * np.log(pn / p)'''

        self.K = ((1 + self.e) / self.k) * p
        self.E = 3 * self.K * (1 - 2 * self.v)
        self.G = self.E / (2 * (1 + self.v))
        self.L = (self.E * self.v) / ((1 + self.v) * (1 - 2 * self.v))
        self.O = (1 + self.e) / (self.l - self.k)

    def check_for_plasticity(self, index: int, stress_tensor: np.ndarray) -> float:

        p = volumetric_scalar(stress_tensor)
        q = deviatoric_scalar(stress_tensor)

        F = (q / self.M)**2 + p * (p - self.pc)

        return F

    def compute_plastic_mulitplier(self, index: int, stress_tensor_trial: np.ndarray) -> float:

        ptrial = volumetric_scalar(stress_tensor_trial)
        qtrial = deviatoric_scalar(stress_tensor_trial)
        pcn = self.pc
        
        # initial guesses for local and sublocal newton-raphson scheme
        dphi = 0 
        pc = self.pc
        
        # iteration scheme parameters
        maxiter    = 1000
        Ftolerance = 1e-6
        Rtolerance = 1e-6

        for i in range(maxiter): # local newton-raphson scheme to calculate plastic multiplier

            for j in range(maxiter): # sublocal newton-raphson scheme to calculate the new value of hardening paramter (pre-consolidation pressure)

                R      = pcn * np.exp(self.O * dphi * ((2 * ptrial - pc) / (1 + 2 * self.K * dphi))) - pc
                Rprime = pcn * np.exp(self.O * dphi * ((2 * ptrial - pc) / (1 + 2 * self.K * dphi))) * ((- self.O * dphi) / (1 + 2 * self.K * dphi)) - 1 

                if abs(R) <= Rtolerance:

                    p = (ptrial + dphi * self.K * pc) / (1 + 2 * self.K * dphi)
                    q = qtrial / (1 + (6 * self.G * dphi) / (self.M**2))
                
                    break
                    
                pc = pc - (R / Rprime)
                
            A = - (self.K * (2 * p - pc)**2) / (1 + (2 * self.K + self.O * pc) * dphi)
            B = - (2 * q**2) / ((self.M**2) * (dphi + ((self.M**2) / (6 * self.G))))
            C = - (p * self.O * pc) * ((2 * p - pc) / (1 + (2 * self.K + self.O * pc) * dphi))

            F = (q**2 / self.M**2) + p * (p - pc)
            Fprime = A + B + C

            if abs(F) <= Ftolerance:

                self.pc = pc    
                return dphi, p, q
                
            dphi = dphi - (F / Fprime)

        raise ValueError('Local Newton-Raphson scheme do not converge!')

    def build_elastoplastic_operator(self, index: int, stress_tensor_converged: np.ndarray, stress_tensor_trial: np.ndarray, yielding_function_value: float, dphi: float) -> np.ndarray:

        delta = np.eye(3)
        identity = np.einsum('ij, kl -> ijkl', delta, delta)
        isotropic = 0.5 * (np.einsum('ik, jl -> ijkl', delta, delta) + np.einsum('il, jk -> ijkl', delta, delta))
        
        if yielding_function_value < 0: # elastoplastic tangent operator degenerates to elastic tangent operator
            
            elastoplastic_modulus = self.L * identity + 2 * self.G * isotropic
            
            return elastoplastic_modulus

        else: # compute elastoplastic tangent operator

            p = volumetric_scalar(stress_tensor_converged)
            q = deviatoric_scalar(stress_tensor_converged)

            Sconve = np.linalg.norm(deviatoric_tensor(stress_tensor_converged), 'fro')
            Strial = np.linalg.norm(deviatoric_tensor(stress_tensor_trial), 'fro')
            xi = Sconve / Strial
  
            n = deviatoric_tensor(stress_tensor_trial) / np.linalg.norm(deviatoric_tensor(stress_tensor_trial))
            
            # pai, pai, por que me abandonastes?

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

            elastoplastic_modulus = 2 * self.G * xi * isotropic \
                                  + (self.K * (a1 + a2 * b1) - (1 / 3) * (2 * self.G) * xi) * identity \
                                  + self.K * (a2 * b2) * np.einsum('ij, kl -> ijkl', delta, n) \
                                  + 2 * self.G * np.sqrt(2 / 3) * (a6 * b1) * np.einsum('ij, kl -> ijkl', n, delta) \
                                  + 2 * self.G * (np.sqrt(2 / 3) * (a5 + a6 * b2) - xi) * np.einsum('ij, kl -> ijkl', n, n)
            
            return elastoplastic_modulus
   






