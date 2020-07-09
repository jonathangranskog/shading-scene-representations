import numpy as np
import pyrr
import os

class Material():
    def __init__(self, color=np.ones(3, dtype=np.float32), emission=np.zeros(3, dtype=np.float32), roughness=1.0, ior=15.0, id=0, texture=None, texture_frequency=np.array([1.0, 1.0])):
        self.color = color
        self.emission = emission
        self.roughness = roughness
        self.ior = ior
        if not texture is None:
            self.texture = os.path.abspath(texture)
        else:
            self.texture = ""
        self.texture_frequency = texture_frequency
        self.id = id

    def as_dict(self):
        d = {}
        d["color"] = self.color.tolist()
        d["emission"] = self.emission.tolist()
        d["roughness"] = self.roughness
        d["ior"] = self.ior
        d["texture"] = self.texture
        d["texture_frequency"] = self.texture_frequency.tolist()
        d["id"] = self.id

        return d