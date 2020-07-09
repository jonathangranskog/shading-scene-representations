import numpy as np
import torch
import torchvision
import os
import sys
import random
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import Dataset
from renderer.interface import RenderInterface
import renderer.randomize.scene_randomizer as sr
import time

def create_dataset(settings, device, iteration=0):
    if settings.cached_dataset:
        train_dataset = PrerenderedDataset(settings, device)
        test_dataset = PrerenderedDataset(settings, device, test=True, iteration=iteration)
    else:
        train_dataset = RTRenderedDataset(settings, device)
        test_dataset = RTRenderedDataset(settings, device, test=True)
    return train_dataset, test_dataset

# A real-time rendered dataset
class RTRenderedDataset():
    def __init__(self, settings, device, test=False):
        self.device = device
        self.scene = settings.dataset

        if not test:
            self.batch_size = settings.batch_size
        else:
            self.batch_size = settings.test_batch_size

        # Set render size
        self.render_size = 0
        self.generator_passes = []
        for i in range(len(settings.model.generators)):
            self.render_size = max(self.render_size, settings.model.generators[i].render_size) 
            gen_passes = settings.model.generators[i].query_passes + settings.model.generators[i].output_passes
            self.generator_passes += gen_passes

        self.observation_passes = []
        for i in range(len(settings.model.representations)):
            self.render_size = max(self.render_size, settings.model.representations[i].render_size) 
            repr_passes = settings.model.representations[i].observation_passes
            self.observation_passes += repr_passes

        # List all the passes used
        self.all_passes = list(set(self.observation_passes + self.generator_passes))
        self.extra_passes = self.all_passes.copy()

        if "beauty" in self.extra_passes: self.extra_passes.remove("beauty")
        if "position" in self.extra_passes: self.extra_passes.remove("position")
        if "normal" in self.extra_passes: self.extra_passes.remove("normal")
        if "depth" in self.extra_passes: self.extra_passes.remove("depth")
        if "albedo" in self.extra_passes: self.extra_passes.remove("albedo")
        if "spec" in self.extra_passes: self.extra_passes.remove("spec")
        if "roughness" in self.extra_passes: self.extra_passes.remove("roughness")
        if "direct" in self.extra_passes: self.extra_passes.remove("direct")
        if "indirect" in self.extra_passes: self.extra_passes.remove("indirect")
        if "specular" in self.extra_passes: self.extra_passes.remove("specular")
        if "diffuse" in self.extra_passes: self.extra_passes.remove("diffuse")
        if "mirror" in self.extra_passes: self.extra_passes.remove("mirror")
        if "mirror_hit" in self.extra_passes: self.extra_passes.remove("mirror_hit")
        if "mirror_normal" in self.extra_passes: self.extra_passes.remove("mirror_normal")

        # Create render interface and other variables
        self.renderer = RenderInterface(self.render_size, settings.seed, device=device.index, scene=self.scene)
        
        self.views_per_scene = settings.views_per_scene
        self.random_num_views = settings.random_num_views
        self.samples = settings.samples_per_pixel
        self.latent_separation = settings.latent_separation

        self.no_camera_anim = False
        self.constant_factor = False

    # Non-buffer passes are checked here
    def extra_pass(self, name, passes):
        # Render image with no NEE
        if name[:5] == "nonee":
            samples = int(name[5:])
            beauty = self.renderer.get_image(samples, nee=False)["beauty"]
            beauty = torch.from_numpy(beauty).float().to(self.device).permute(2, 0, 1)
            beauty = beauty.reshape((1, 1, *beauty.shape))
            return beauty
        # Create a noisy image 
        if name == "noise":
            strength = 0.25
            tmp = passes["beauty"].clone()
            noise_level = torch.ones_like(tmp)[:, :, 0]
            noise_level -= strength
            snp_noise = D.Bernoulli(probs=noise_level).sample()
            tmp = D.Normal(tmp, 2 * strength).sample()
            tmp = snp_noise * tmp
            return tmp
        # Create a gray scale image
        if name == "gray":
            tmp = passes["beauty"].clone()
            tmp = torch.mean(tmp, dim=2, keepdim=True)
            return tmp


    # Creates a dictionary of all available passes
    def handle_passes(self):
        passes = self.renderer.get_image(self.samples)

        # Convert to PyTorch tensors
        for p in passes:
            passes[p] = torch.from_numpy(passes[p]).float().to(self.device).permute(2, 0, 1)
            passes[p] = passes[p].reshape((1, 1, *passes[p].shape))

        # Create all of the requested extra passes
        for p in self.extra_passes:
            extra = self.extra_pass(p, passes)
            if extra is not None:
                passes[p] = extra

        return passes

    def __iter__(self):
        return self

    def __next__(self):
        if not self.random_num_views:
            self.num_observations = self.views_per_scene - 1
        else:
            self.num_observations = random.randint(1, self.views_per_scene)

        self.renderer.random_scene()
        self.renderer.random_view()

        if self.latent_separation and not self.constant_factor:
            self.factor = random.randint(0, 2)
        elif not self.constant_factor:
            self.factor = -1

        if self.no_camera_anim:
            views = []
            for i in range(self.views_per_scene):
                views.append(random.random())

        for b in range(self.batch_size):
            # Generate random variation either only one aspect if partitioned
            # Else randomize entire scene
            if self.factor == 0:
                self.renderer.randomize_lighting()
            elif self.factor == 1:
                    self.renderer.randomize_geometry()
            elif self.factor == 2:
                    self.renderer.randomize_materials()
            else:
                self.renderer.random_scene()

            # Create observations of scene
            for i in range(self.num_observations + 1):
                if not self.no_camera_anim:
                    self.renderer.random_view()
                else:
                    self.renderer.animate_view(views[i])

                passes = self.handle_passes()
                pose = torch.from_numpy(self.renderer.get_pose()).float().to(self.device)
                pose = pose.reshape(1, 1, pose.shape[0], 1, 1)

                if i == 0:
                    batch_passes = passes
                    poses = pose
                else:
                    poses = torch.cat([poses, pose], dim=1)
                    for key in passes:
                        batch_passes[key] = torch.cat([batch_passes[key], passes[key]], dim=1)

            if b == 0:
                all_passes = batch_passes
                all_poses = poses
            else:
                all_poses = torch.cat([all_poses, poses], dim=0)
                for key in batch_passes:
                    all_passes[key] = torch.cat([all_passes[key], batch_passes[key]], dim=0)
        
        # Get passes for observations
        observations = {}
        for key in self.all_passes:
            observations[key] = all_passes[key][:, :-1].contiguous()
            
        # Get passes for queries
        queries = {}
        for key in self.all_passes:
            queries[key] = all_passes[key][:, -1].contiguous()

        data = {}
        data["observation_images"] = observations
        data["query_images"] = queries
        data["observation_poses"] = all_poses[:, :-1].contiguous()
        data["query_poses"] = all_poses[:, -1].contiguous()
        data["all_passes"] = all_passes
        data["factor"] = self.factor

        return data

    # Helper function to get passes if animating camera
    def get_current_view(self):
        # Requires that one iteration has been performed.
        passes = self.handle_passes()
        pose = torch.from_numpy(self.renderer.get_pose()).float().to(self.device)
        pose = pose.reshape(1, pose.shape[0], 1, 1)

        # Get passes for queries
        queries = {}
        for key in self.all_passes:
            queries[key] = passes[key][:, -1]
            
        return (queries, pose)

# Load dataset from disk
class PrerenderedDataset():
    def __init__(self, settings, device, test=False, iteration=0):
        self.device = device
        
        if not test:
            self.dir = settings.train_data_dir
            self.batch_size = settings.batch_size
        else:
            self.dir = settings.test_data_dir
            self.batch_size = settings.test_batch_size

        self.files = sorted([f for f in os.listdir(self.dir) if "npz" in f])

        self.observation_passes = []
        self.generation_passes = []

        for i in range(len(settings.model.representations)):
            repr_passes = settings.model.representations[i].observation_passes
            self.observation_passes = self.observation_passes + repr_passes

        for i in range(len(settings.model.generators)):
            gen_passes = settings.model.generators[i].query_passes + settings.model.generators[i].output_passes
            self.generation_passes = self.generation_passes + gen_passes

        self.all_passes = self.generation_passes + self.observation_passes
        self.all_passes = list(set(self.all_passes))

        self.views_per_scene = settings.views_per_scene
        self.random_num_views = settings.random_num_views
        self.samples = settings.samples_per_pixel
        self.test = test

        self.factor = -1
        self.index = iteration

    def __iter__(self):
        return self

    def __next__(self): 
        # Some of the batches give errors when loading
        # Most likely due to some data transfer error when moving from cluster to desktop
        attempts = 0
        while attempts < 25:
            if self.test:
                self.index = (self.index + 1) % len(self.files)
                idx = self.index
            else:
                idx = random.randint(0, len(self.files) - 1)
            data_file = os.path.join(self.dir, self.files[idx])
            
            try:
                data = np.load(data_file)
                break
            except:
                continue
                
        
        poses = data['poses'][:self.batch_size]
        if len(self.files) != 1 and not self.test:
            observation_indices = np.random.permutation(poses.shape[1])
            poses = poses[:, observation_indices]

        if 'factor' in data:
            self.factor = data['factor']
        else:
            self.factor = -1

        passes = {}
        for key in self.all_passes:
            passes[key] = torch.from_numpy(np.nan_to_num(data[key])).float().to(self.device)[:self.batch_size]
            if "indirect" in key:
                passes[key] = passes[key].clamp(0, 1e25)
            if len(self.files) != 1 and not self.test:
                passes[key] = passes[key][:, observation_indices]

        # Get passes for observations
        observations = {}
        for key in self.all_passes:
            observations[key] = passes[key][:, :-1].contiguous()
            
        # Get passes for queries
        queries = {}
        for key in self.all_passes:
            queries[key] = passes[key][:, -1].contiguous()

        poses = torch.from_numpy(poses).float().to(self.device)

        data = {}
        data["observation_images"] = observations
        data["query_images"] = queries
        data["observation_poses"] = poses[:, :-1].contiguous()
        data["query_poses"] = poses[:, -1].contiguous()
        data["all_passes"] = passes
        data["factor"] = self.factor
        return data

def get_buffer_length(buffer):
    scalar_buffers = ["spec", "roughness", "depth", "ao", "shadows", "id"]
    if buffer in scalar_buffers:
        return 1
    else:
        return 3
