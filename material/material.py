class SoilMaterialType:

    def __init__(self):
        
        self.name: str
        self.v: float = 0.35
        self.e: float = 1
        self.internal_friction_angle : float 
        self.pc: float = 150
        self.M: float = 1.20
        self.k: float = 0.01
        self.l: float = 0.10
    
class StealMaterialType:

    def __init__(self):
        
        self.name: str


class ConcreteMaterialType:
    def __init__(self):
        
        self.name: str



argila = SoilMaterialType()

        

