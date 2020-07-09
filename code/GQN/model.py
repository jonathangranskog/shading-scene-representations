import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from GQN.representation import TowerNet, PyramidNet
from GQN.generator import GQNGenerator, UnetGenerator
from GQN.handlers import GeneratorHandler, RepresentationHandler
from tqdm import tqdm

"""
The full GQN model, containing the representation and generation networks
"""
class GenerativeQueryNetwork(nn.Module):
    def __init__(self, settings, iteration):
        super(GenerativeQueryNetwork, self).__init__()
        
        self.generator = GeneratorHandler(settings)
        self.representation = RepresentationHandler(settings, iteration)
        
        representation_params = sum(p.numel() for p in self.representation.parameters() if p.requires_grad)
        print("Representation: " + str(representation_params) + " trainable parameters.")

        generator_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        print("Generator: " + str(generator_params) + " trainable parameters.")

    def forward(self, data, factor, iteration, representations=None, additional=None):
        observation_images = data["observation_images"]
        observation_poses = data["observation_poses"]
        query_passes = data["query_images"]
        query_poses = data["query_poses"]

        if representations is None:
            representations, m, r_loss = self.representation(observation_images, query_passes, observation_poses, query_poses, factor)
        y, gen_loss, metrics = self.generator(observation_images, observation_poses, query_passes, query_poses, representations, iteration)

        for key in m:
            metrics[key] = m[key]

        gen_loss += r_loss

        return y, gen_loss, metrics

    def sample(self, data, representations=None):
        observation_images = data["observation_images"]
        observation_poses = data["observation_poses"]
        query_passes = data["query_images"]
        query_poses = data["query_poses"]
        
        if representations is None:
            representations, _, _ = self.representation(observation_images, query_passes, observation_poses, query_poses, -1)
        y = self.generator.sample(observation_images, observation_poses, query_passes, query_poses, representations)
        return y

    def compute_representations(self, data, factor=-1):
        observation_images = data["observation_images"]
        observation_poses = data["observation_poses"]
        query_passes = data["query_images"]
        query_poses = data["query_poses"]
        
        return self.representation(observation_images, query_passes, observation_poses, query_poses, -1)

    
    # Ugly function that splits the sampling into 64x64 tiles to handle larger memory issues
    # Also disables gradient computation. This is used to generate larger figures.
    # assumes pixel network for proper results.
    def tiled_sample(self, data, representations=None):
        observation_images = data["observation_images"]
        observation_poses = data["observation_poses"]
        query_passes = data["query_images"]
        query_poses = data["query_poses"]

        with torch.no_grad():
            if representations is None:
                representations, _, _ = self.representation(observation_images, query_passes, observation_poses, query_poses, -1)
            
            first_key = list(query_passes.keys())[0]
            b, c, h, w = query_passes[first_key].shape
            h_it = h // 64
            w_it = w // 64
                
            for i in tqdm(range(w_it)):
                for j in range(h_it):
                    qpasses = {}
                    for key in query_passes:
                        qpasses[key] = query_passes[key][:, :, i*64:(i + 1)*64, j*64:(j + 1)*64]

                    y = self.generator.sample(observation_images, observation_poses, qpasses, query_poses, representations)
            
                    for g in range(len(y)):
                        for key in y[g]:
                            b, c, nh, nw = y[g][key].shape
                            y[g][key] = y[g][key].view(b, c, nh, nw)

                    if j == 0:
                        Y = y
                    else:
                        for g in range(len(y)):
                            for key in y[g]:
                                Y[g][key] = torch.cat([Y[g][key], y[g][key]], dim=3)

                if i == 0:
                    YY = Y
                else:
                    for g in range(len(y)):
                        for key in Y[g]:
                            YY[g][key] = torch.cat([YY[g][key], Y[g][key]], dim=2)

        return YY