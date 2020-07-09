import numpy as np
import numpy

class Camera():
    def __init__(self, position, lookat, fov, near, far):
        self.position = position
        self.lookat = lookat
        self.fov = fov
        self.near = near
        self.far = far

