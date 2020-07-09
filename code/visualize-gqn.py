import torch
import torchvision.transforms as transforms
import numpy as np
import os
import math
import random
import time
import pyrr
import sys
import copy

from GQN.model import GenerativeQueryNetwork
from argparse import ArgumentParser
from util.datasets import RTRenderedDataset
from util.config import configure, read_checkpoint
from util.settings import *
from PIL import Image
from renderer.interface import RenderInterface

'''
This script lets you walk around scenes and visualize network predictions
'''

parser = ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to load')
parser.add_argument('--config_dir', type=str, default='', help='Where config file is located')
parser.add_argument('--config', type=str, default='', help='Which config to read')
parser.add_argument('--device', type=str, default='', help='Device to run on')
parser.add_argument('--find_checkpoints', action='store_true', help='Attempt to find matching checkpoints automatically')
parser.add_argument('--scene_file', type=str, default='')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

settings = configure(args, ignore_data=True)
checkpoint, iteration = read_checkpoint(args, settings)

# This is a trick to use the DataLoader instead of creating everything from scratch
settings.batch_size = 1
samples = settings.samples_per_pixel
settings.samples_per_pixel = 1
dataset = RTRenderedDataset(settings, device)
iterator = iter(dataset)
t = 0.0
data = next(iterator)
dataset.samples = samples
observation_samples = 256

def init_data():
    global data, dataset
    dataset.samples = samples
    queries = dataset.get_current_view()
    data["query_images"] = queries[0]
    data["query_poses"] = queries[1]
    random_observations()

def random_scene():
    global data, iterator
    dataset.renderer.random_scene()
    init_data()

def set_spp(num_samples):
    global dataset
    dataset.samples = num_samples

def random_observations():
    global data, dataset
    # Create observations manually
    dataset.samples = observation_samples
    dataset.renderer.random_view()
    view1 = dataset.get_current_view()
    dataset.renderer.random_view()
    view2 = dataset.get_current_view()
    dataset.renderer.random_view()
    view3 = dataset.get_current_view()
    for key in data["observation_images"].keys():
        data["observation_images"][key][0][0] = view1[0][key][0]
        data["observation_images"][key][0][1] = view2[0][key][0]
        data["observation_images"][key][0][2] = view3[0][key][0]
    data["observation_poses"][0][0] = view1[1][0]
    data["observation_poses"][0][1] = view2[1][0]
    data["observation_poses"][0][2] = view3[1][0]
    dataset.samples = samples

if args.scene_file != '':
    if not os.path.isfile(args.scene_file):
        print("Provided scene file does not exist!")
        quit()

    dataset.renderer.load_scene_file(args.scene_file)
    init_data()
else:
    random_scene()
    
iteration = checkpoint['iteration']

# Create network
net = GenerativeQueryNetwork(settings, iteration)
if 'representation_state' in checkpoint and 'generator_state' in checkpoint:
    net.representation.load_state_dict(checkpoint['representation_state'])
    net.generator.load_state_dict(checkpoint['generator_state'])
else:
    net.load_state_dict(checkpoint['model_state'])
    for i in range(len(net.representation.representations)):
        net.representation.representations[i].iteration = iteration
net = net.to(device)
net.eval()

print(settings)

def format_buffer(buf):
    tmp = buf.clone()
    if tmp.shape[0] == 1:
        tmp = tmp.repeat(3, 1, 1)
    return tmp.detach().cpu().permute(1, 2, 0) ** (1 / 2.2)

import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from moderngl_window.scene.camera import KeyboardCamera
import imgui


class WindowEvents(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1200, 600)
    aspect_ratio = window_size[0] / window_size[1]
    title = "Neural Renderer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.space_down = False

        self.prog = self.ctx.program(vertex_shader="""
        #version 330
        uniform mat4 transform;
        uniform vec3 clr;
        uniform float aspect;

        in vec3 in_vert;
        out vec3 color;
        out vec2 uv;
        void main() {
            vec3 pos = vec3(in_vert.x, in_vert.y * aspect, 0.0);
            gl_Position = transform * vec4(pos, 1.0);
            uv = vec2(in_vert.x * 0.5 + 0.5, in_vert.y * 0.5 + 0.5);
            uv.y = 1.0 - uv.y;
            color = vec3(1, 0, 0);
        }
        """,
        fragment_shader="""
        #version 330
        uniform sampler2D tex_sampler;

        out vec4 fragColor;
        in vec3 color;
        in vec2 uv;
        void main() {
            fragColor = vec4(texture(tex_sampler, uv).rgb, 1.0);
        }
        """,
        )

        self.reference_texture = self.ctx.texture((dataset.render_size, dataset.render_size), components=3)
        self.prediction_texture = self.ctx.texture((dataset.render_size, dataset.render_size), components=3)
        self.reference_texture.repeat_x = False
        self.reference_texture.repeat_y = False
        self.prediction_texture.repeat_x = False
        self.prediction_texture.repeat_y = False
        self.reference_texture.use(5)
        self.prediction_texture.use(6)

        self.prog['aspect'].value = 12 / 6
        T = pyrr.matrix44.create_from_translation(np.array([-0.5, 0.15, 0]))
        T2 = pyrr.matrix44.create_from_translation(np.array([0.5, 0.15, 0]))
        S = pyrr.matrix44.create_from_scale(np.array([0.4, 0.4, 1]))
        self.M = pyrr.matrix44.multiply(S, T)
        self.M2 = pyrr.matrix44.multiply(S, T2)
        self.transform = self.prog['transform']
        self.transform.value = tuple(self.M.flatten())
        
        self.observation_textures = []
        self.observation_transforms = []
        for i in range(1, settings.views_per_scene):
            self.observation_textures.append(self.ctx.texture((dataset.render_size, dataset.render_size), components=3))
            self.observation_textures[-1].repeat_x = False
            self.observation_textures[-1].repeat_y = False
            self.observation_textures[-1].use(6 + i)

            T = pyrr.matrix44.create_from_translation(np.array([-0.825 + (i - 1) * 0.165, -0.825, 0]))
            S = pyrr.matrix44.create_from_scale(np.array([0.075, 0.075, 1]))
            M = pyrr.matrix44.multiply(S, T)
            self.observation_transforms.append(M)

        self.buffer_textures = []
        self.buffer_transforms = []
        for i in range(len(settings.model.generators[-1].query_passes)):
            self.buffer_textures.append(self.ctx.texture((dataset.render_size, dataset.render_size), components=3))
            self.buffer_textures[-1].repeat_x = False
            self.buffer_textures[-1].repeat_y = False
            self.buffer_textures[-1].use(6 + settings.views_per_scene + i)

            T = pyrr.matrix44.create_from_translation(np.array([0.175 + i * 0.165, -0.825, 0]))
            S = pyrr.matrix44.create_from_scale(np.array([0.075, 0.075, 1]))
            M = pyrr.matrix44.multiply(S, T)
            self.buffer_transforms.append(M)
        
        self.camera = KeyboardCamera(self.wnd.keys, 45.0, 1.0, 0.1, 100.0)
        self.camera.position[0] = 1.5
        self.camera.position[1] = 1.5
        self.camera.position[2] = -1.5
        self.camera._velocity = -2.5
        self.camera._mouse_sensitivity = -0.1

        self.quad = np.array([-1.0, -1.0, 0.0,
                            1.0, -1.0, 0.0,
                            1.0,  1.0, 0.0,
                            -1.0, -1.0, 0.0,
                            1.0,  1.0, 0.0,
                            -1.0,  1.0, 0.0], dtype='f4')
        self.quad_vao = self.ctx.simple_vertex_array(self.prog, self.ctx.buffer(self.quad), 'in_vert')

    def render(self, time: float, frametime: float):
        # Load transform
        view = self.camera.matrix
        view_inverse = pyrr.matrix44.inverse(view)
        position = pyrr.matrix44.apply_to_vector(view_inverse, np.array([0.0, 0.0, 0.0, 1.0]))
        lookat = pyrr.matrix44.apply_to_vector(view_inverse, np.array([0.0, 0.0, 1.0, 1.0]))
        dataset.renderer.renderer.set_camera(position[:3], lookat[:3])
        
        # Get reference and draw
        dataset.samples = samples
        queries = dataset.get_current_view()
        data["query_images"] = queries[0]
        data["query_poses"] = queries[1]

        reference = format_buffer(data["query_images"][settings.model.output_pass][0])
        reference = reference.clamp(0, 1).numpy()
        reference = (reference * 255).astype(np.uint8)
        self.reference_texture.write(reference.tobytes())
        self.prog['tex_sampler'].value = 5
        self.transform.value = tuple(self.M.flatten())
        self.quad_vao.render(mode=moderngl.TRIANGLES)

        # Draw observations
        for i in range(len(self.observation_textures)):
            observation = format_buffer(data["observation_images"][settings.model.output_pass][0][i])
            observation = observation.clamp(0, 1).numpy()
            observation = (observation * 255).astype(np.uint8)
            self.observation_textures[i].write(observation.tobytes())
            self.prog['tex_sampler'].value = 6 + 1 + i
            self.transform.value = tuple(self.observation_transforms[i].flatten())
            self.quad_vao.render(mode=moderngl.TRIANGLES)

        # Draw G-buffer (TODO)
        for i in range(len(self.buffer_textures)):
            buffer = format_buffer(data["query_images"][settings.model.generators[-1].query_passes[i]][0])
            buffer = buffer.clamp(0, 1).numpy()
            buffer = (buffer * 255).astype(np.uint8)
            self.buffer_textures[i].write(buffer.tobytes())
            self.prog['tex_sampler'].value = 6 + settings.views_per_scene + i
            self.transform.value = tuple(self.buffer_transforms[i].flatten())
            self.quad_vao.render(mode=moderngl.TRIANGLES)

        # Network sample and draw
        prediction = net.sample(data)
        pred = format_buffer(prediction[-1][settings.model.output_pass][0])
        pred = pred.clamp(0, 1).numpy()
        pred = (pred * 255).astype(np.uint8)
        self.prediction_texture.write(pred.tobytes())
        self.prog['tex_sampler'].value = 6
        self.transform.value = tuple(self.M2.flatten())
        self.quad_vao.render(mode=moderngl.TRIANGLES)

        self.render_ui()

    def render_ui(self):
        global samples, observation_samples
        imgui.new_frame()

        imgui.begin("Options", True)

        if imgui.button("Random Scene"):
            random_scene()

        if imgui.button("Randomize Observations"):
            random_observations()

        _, samples = imgui.drag_int("Query SPP", samples, min_value=1, max_value=1024)
        _, observation_samples = imgui.drag_int("Observation SPP", observation_samples, min_value=1, max_value=1024)

        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def key_event(self, key, action, modifiers):
        global samples, observation_samples
        self.imgui.key_event(key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.space_down = True
            if key == self.wnd.keys.R:
                random_scene()
            if key == self.wnd.keys.O:
                random_observations()
            if key == self.wnd.keys.J:
                samples += 10
            if key == self.wnd.keys.H:
                samples = max(1, samples - 10)
            if key == self.wnd.keys.M:
                observation_samples += 10
            if key == self.wnd.keys.N:
                observation_samples = max(1, observation_samples - 10)

        if action == self.wnd.keys.ACTION_RELEASE:
            if key == self.wnd.keys.SPACE:
                self.space_down = False
                imgui.set_window_focus()

        if self.space_down:
            self.camera.key_input(key, action, modifiers)

    def mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)
        if self.space_down:
            self.camera.rot_state(dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)


sys.argv[1:] = []
mglw.run_window_config(WindowEvents)