class SoilMaterialType:

    def __init__(self):
        
        self.variety: str = "Pure Clay" # short description of the clay
        self.v: float = 0.35 # poisson ratio [-]
        self.e: float = 1.00 # void ratio [-]
        self.phi : float = 0 # internal friction angle [rad]
        self.pc: float = 150 # initial pre-consolidation pressure [kPa]
        self.M: float = 1.20 # critial state line slope [-]
        self.k: float = 0.01 # swelling consolidation line slope [-]
        self.l: float = 0.10 # normal consolidation line slope [-]
        
argila = SoilMaterialType()

        

