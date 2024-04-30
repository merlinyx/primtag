class Material:
    def __init__(self, density, name="Unknown"):
        self.name = name
        self.density = density
    
    def __str__(self):
        return self.name
    
    def get_mass(self, volume):
        return self.density * volume
