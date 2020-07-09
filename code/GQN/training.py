import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
import os
import math
import time
import pytorch_ssim

from GQN.model import GenerativeQueryNetwork
from torch.utils.data import DataLoader

# Class that contains networks and optimizers
class NetworkTrainer():
    def __init__(self, settings, device, train_dataset, test_dataset, checkpoint=None):
        # Check if checkpoint load requested
        self.device = device

        if checkpoint is None:
            self.iteration = 1
            self.net = GenerativeQueryNetwork(settings, self.iteration).to(self.device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        else:
            # Only load representation network from settings
            self.iteration = checkpoint['iteration']
            self.net = GenerativeQueryNetwork(settings, self.iteration)

            if 'representation_state' in checkpoint and 'generator_state' in checkpoint:
                self.net.representation.load_state_dict(checkpoint['representation_state'])
                self.net.generator.load_state_dict(checkpoint['generator_state'])
            else:
                self.net.load_state_dict(checkpoint['model_state'])
            self.net = self.net.to(self.device)

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.net.eval()

        if torch.cuda.device_count() > 1:
            print("Using " + str(torch.cuda.device_count()) + " GPUs!")

        self.test_iterator = iter(test_dataset)
        self.train_iterator = iter(train_dataset)

        self.output_pass = settings.model.output_pass

    def step(self, i):
        self.optimizer.zero_grad()

        # Get training batch
        data = next(self.train_iterator)
        factor = data["factor"]
        del data["factor"]

        # Run through network
        image, loss, metrics = self.net(data, factor, i)
        
        # Backpropagate loss and take gradient step
        self.net.representation.retain_gradients()
        loss.backward()

        self.net.representation.correct_latent_gradients(factor)
        self.optimizer.step()
        
        with torch.no_grad():
            # Compute SSIM
            metrics["total_loss"] = loss.item()

        del loss
        return metrics
