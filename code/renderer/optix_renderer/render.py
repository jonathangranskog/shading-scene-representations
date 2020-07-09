import numpy as np
import random
import os
import renderer.optix_renderer.build.lib.rnd as rnd

RND_CONTEXT = 0

# This class accesses the C++ functions for the Optix renderer
class OptixRenderer():
    def __init__(self, size, device):
        global RND_CONTEXT
        self.size = size
        self.device = device
        if RND_CONTEXT == 0:
            RND_CONTEXT = rnd.Context(self.size, device)
    
    def load_scene_file(self, filename):
        fname = os.path.abspath(filename)
        RND_CONTEXT.load_scene_file(fname)

    def load_scene_json(self, desc):
        RND_CONTEXT.load_scene_json(desc)

    def get_pose(self):
        view_matrix = RND_CONTEXT.get_view_matrix()
        view_matrix = np.asanyarray(view_matrix)
        pose = view_matrix.reshape(16)
        return pose

    def set_camera(self, pos, lookat):
        px = pos[0]
        py = pos[1]
        pz = pos[2]
        lx = lookat[0]
        ly = lookat[1]
        lz = lookat[2]
        RND_CONTEXT.set_camera(px, py, pz, lx, ly, lz)

    def set_nee(self, nee):
        RND_CONTEXT.set_nee(nee)

    def draw_scene_to_tensor(self, samples):
        RND_CONTEXT.render(samples)
        image = np.asanyarray(RND_CONTEXT.get_image())
        return image

    def draw_scene(self, samples):
        tensor = np.flipud(self.draw_scene_to_tensor(samples)).copy()
        passes = {}
        passes["beauty"] = np.nan_to_num(tensor[:, :, 0:3])
        passes["normal"] = tensor[:, :, 3:6]
        passes["depth"] = tensor[:, :, 6:7]
        passes["position"] = tensor[:, :, 7:10]
        passes["albedo"] = tensor[:, :, 10:13]
        passes["roughness"] = tensor[:, :, 13:14]
        passes["direct"] = np.nan_to_num(tensor[:, :, 14:17])
        passes["indirect"] = passes["beauty"] - passes["direct"]
        passes["diffuse"] = tensor[:, :, 17:20]
        passes["specular"] = passes["beauty"] - passes["diffuse"]
        passes["mirror"] = tensor[:, :, 20:23]
        passes["mirror_hit"] = tensor[:, :, 23:26]
        passes["mirror_normal"] = tensor[:, :, 26:29]
        passes["shadows"] = tensor[:, :, 29:30]
        passes["ao"] = tensor[:, :, 30:31]
        passes["id"] = tensor[:, :, 31:34]
        return passes