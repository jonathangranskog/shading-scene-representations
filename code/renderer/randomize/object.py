import numpy as np
import pyrr
import random
import os

class Object():
    def __init__(self, transform=None, material=None):

        if transform is None:
            self.transform = pyrr.matrix44.create_identity(dtype=np.float32)
        else:
            self.transform = transform

        if material is None:
            self.material = ""
        else:
            self.material = material
        
        self.visible = True

    def as_dict(self):
        d = {}
        transform = self.transform.tolist()
        material = self.material
        d["transform"] = transform
        d["material"] = material
        return d

class Grid(Object):
    def __init__(self, position=np.zeros(3, dtype=np.float32), normal=np.array([0, 1, 0], dtype=np.float32), size=np.ones(3, dtype=np.float32), transform=None, material=None):
        super(Grid, self).__init__(transform, material)
        self.type = "grid"
        self.position = position
        self.normal = normal
        self.size = size

    def as_dict(self):
        d = super(Grid, self).as_dict()
        d["type"] = self.type
        d["position"] = self.position.tolist()
        d["normal"] = self.normal.tolist()
        d["size"] = self.size.tolist()
        return d

class File(Object):
    def __init__(self, filename, transform=None, material=None):
        super(File, self).__init__(transform, material)
        self.filename = os.path.abspath(filename)
        self.type = "file"
    
    def as_dict(self):
        d = super(File, self).as_dict()
        d["type"] = self.type
        d["filename"] = self.filename
        return d

class Sphere(Object):
    def __init__(self, position, radius, transform=None, material=None):
        super(Sphere, self).__init__(transform, material)
        self.position = position
        self.radius = radius
        self.type = "sphere"

    def as_dict(self):
        d = super(Sphere, self).as_dict()
        d["type"] = self.type
        d["radius"] = self.radius
        d["position"] = self.position.tolist()
        return d