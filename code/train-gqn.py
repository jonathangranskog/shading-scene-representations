import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
import os
import math
import random
import time
import pytorch_ssim
import gc
import copy

'''
This is the main training script that is called to train GQNs
Give it a config file and it will train the network
'''

from argparse import ArgumentParser
from util.datasets import create_dataset
from util.config import configure, read_checkpoint
from GQN.training import NetworkTrainer
from tensorboardX import SummaryWriter

parser = ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to load')
parser.add_argument('--device', type=str, default='', help='Device to run on')
parser.add_argument('--config_dir', type=str, default='', help='Where config file is located')
parser.add_argument('--config', type=str, default='', help='Which config to read')
parser.add_argument('--out_folder', type=str, default='../checkpoints/', help='Where to save checkpoints')
args = parser.parse_args()

os.environ['TORCH_HOME'] = ''

cuda = torch.cuda.is_available()
if args.device is not '':
    device = torch.device(args.device)
else:
    device = torch.device("cuda:0" if cuda else "cpu")

# Set up directory paths, seed and checkpoint
settings = configure(args)
checkpoint, iteration = read_checkpoint(args, settings)

# Create a torch Dataset based on selected dataset
train_dataset, test_dataset = create_dataset(settings, device, iteration=iteration)

# Create network
trainer = NetworkTrainer(settings, device, train_dataset, test_dataset, checkpoint=checkpoint)

print(settings)

# Prepare checkpoint folders
if args.checkpoint == '':
    checkpoint_dir = os.path.abspath(args.out_folder)
else:
    checkpoint_dir = os.path.dirname(args.checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)

# Create tensorboard writer
if settings.logging:
    logdir = "../runs/" + settings.job_name
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir=logdir)

for i in range(iteration, settings.iterations + 1):
    # Perform one training step
    t0 = time.time()
    metrics = trainer.step(i)
    t1 = time.time()

    print("Iteration " + str(i) + " completed. Loss: " + str(metrics["total_loss"]) + " Time: " + str(t1 - t0)[0:8])

    # Save checkpoint
    if i % settings.checkpoint_interval == 0:
        state_dict = {'iteration' : i + 1,
                        'model_state' : trainer.net.state_dict(),
                        'optimizer_state' : trainer.optimizer.state_dict()}
        ckpt_path = os.path.join(checkpoint_dir, settings.job_name + '_%d.pt' % i)
        print("Saving checkpoint to: " + ckpt_path)
        torch.save(state_dict, ckpt_path)

    # Log
    if settings.logging and i % settings.test_interval == 0:
        for key, value in metrics.items():
            writer.add_scalar('data/' + key, value, i)

        with torch.no_grad():
            data = next(trainer.test_iterator)
            generated_buffers = trainer.net.sample(data)

            for g in range(len(generated_buffers)):
                for key in generated_buffers[g]:
                    ssim = pytorch_ssim.ssim(data["query_images"][key], generated_buffers[g][key]).mean().item()
                    writer.add_scalar('data/ssim_' + key + str(g), ssim, i)