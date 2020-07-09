import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from renderer.optix_renderer.render import OptixRenderer
import renderer.randomize.scene_randomizer as sr
        
'''
This is the render interface file that communicates with the rasterizer and path tracer
The PyTorch DataReader has access to a RenderInterface class
'''

class RenderInterface():
    def __init__(self, size, seed, device=0, scene="room", hidden=True):
        self.size = size
        self.nee = True
        self.renderer = OptixRenderer(size, device)
        self.randomizer = sr.select_randomizer(scene, seed)

        self.random_scene()
        self.random_view()
        self.scene_updated = True

        if not hidden:
            self.createWindow()

    def get_pose(self):
        pose = self.renderer.get_pose()
        return pose

    # Loads a scene from the randomizer
    def load_scene(self):
        params = self.randomizer.generate_params()
        json_string = self.randomizer.get_json(params)
        self.renderer.load_scene_json(json_string)

    def load_scene_file(self, json_file):
        self.renderer.load_scene_file(json_file)

    def random_scene(self):
        self.randomizer.random_scene()
        self.scene_updated = True

    def randomize_lighting(self):
        self.randomizer.randomize_lighting()
        self.scene_updated = True

    def randomize_geometry(self):
        self.randomizer.randomize_geometry()
        self.scene_updated = True

    def randomize_materials(self):
        self.randomizer.randomize_materials()
        self.scene_updated = True

    def random_view(self):
        self.randomizer.random_view()
        self.renderer.set_camera(self.randomizer.camera.position, self.randomizer.camera.lookat)
        
    def animate_view(self, t):
        self.randomizer.animate_view(t)
        self.renderer.set_camera(self.randomizer.camera.position, self.randomizer.camera.lookat)

    # Sets up the output buffer for the datareader
    def get_image(self, samples=1, nee=True):
        if self.scene_updated:
            self.load_scene()
            self.scene_updated = False

        self.renderer.set_nee(nee)
        passes = self.renderer.draw_scene(samples)
        return passes

    def set_device(self, device):
        self.renderer.set_device(device)

    # Helper function to visualize scenes using the test_renderer script
    def draw(self, *args):
        self.frame = self.frame % 100
        if self.frame % 10 == 0:
            self.random_scene()
            
        self.animate_view(self.frame * 0.01)
        render = self.get_image(samples=16, nee=True)
        tensor = render["beauty"] 
        self.img.set_data(tensor ** (1/2.2))
        self.img2.set_data((render["normal"] + 1) * 0.5 ** (1/2.2))
        self.frame += 1
    
    # Helper function to visualize scenes using the test_renderer script
    def createWindow(self):
        self.frame = 0
        self.fig = plt.figure(figsize=(8, 4))
        self.fig.add_subplot(1, 2, 1)
        plt.axis('off')
        self.img = plt.imshow(np.zeros((self.size, self.size, 3)), animated=True)
        self.fig.add_subplot(1, 2, 2)
        plt.axis('off')
        self.fig.tight_layout()
        self.img2 = plt.imshow(np.zeros((self.size, self.size, 3)), animated=True)
        ani = animation.FuncAnimation(self.fig, self.draw, frames=99, interval=100)

        plt.show()
