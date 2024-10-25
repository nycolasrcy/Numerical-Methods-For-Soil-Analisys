import numpy as np

def I1(tensor: np.ndarray) -> float:
    return np.trace(tensor)

def I2(tensor: np.ndarray) -> float:
    return 0.5 * (np.trace(tensor)**2 - np.trace(tensor**2))

def I3(tensor: np.ndarray) -> float:
    return np.linalg.det(tensor)

def volumetric_scalar(tensor: np.ndarray) -> float:
    return I1(tensor)/3

def deviatoric_scalar(tensor: np.ndarray)-> float:
    return np.sqrt(3 * ( (1/3) * (I1(tensor)**2) - I2(tensor) ) )

def volumetric_tensor(tensor: np.ndarray) -> np.ndarray:
    return volumetric_scalar(tensor) * np.eye(3)

def deviatoric_tensor(tensor: np.ndarray) -> np.ndarray:
    return tensor - volumetric_tensor(tensor)