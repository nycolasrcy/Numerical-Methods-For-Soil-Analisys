


class CapModel:

    def __init__(self,material):

    def update_state_variable(self, index: int, stress_tensor_converged: np.ndarray):
    
    def check_for_plasticity(self, index: int, stress_tensor: np.ndarray) -> float:

    def compute_plastic_multiplier(self, index: int, stress_tensor_trial: np.ndarray) -> float:

    def build_elastoplastic_operator(self, index: int, stress_tensor_converged: np.ndarray, stress_tensor_trial: np.ndarray, dphi: float) -> np.ndarray: