import numpy as np
import os
import math
import random

from argparse import ArgumentParser
from util.config import configure
from tqdm import tqdm

import renderer.randomize.scene_randomizer as sr

'''
This script creates a folder structure containing 'size' + 'testing_size' batches,
where each batch consists of a certain number of scenes based on the batch size.
Each scene also consists of N rendered views. 
This script does not generate any imagery by itself, only the scene descriptions required to
render it.
'''

parser = ArgumentParser()
parser.add_argument('--config_dir', type=str, default='', help='Where config file is located')
parser.add_argument('--config', type=str, default='', help='Config file to read')
parser.add_argument('--size', type=int, default=9000, help='How many batches to include in dataset')
parser.add_argument('--testing_size', type=int, default=1000, help='How many testing batches to include in dataset')
parser.add_argument('--device', type=str, default='', help='Which device to run on')
parser.add_argument('--find_checkpoints', action='store_true', help='Attempt to find checkpoints automatically')
parser.add_argument('--out_folder', type=str, default='tmp/', help='Folder to save JSON files to')
args = parser.parse_args()

settings = configure(args, ignore_data=True)
randomizer = sr.select_randomizer(settings.dataset, settings.seed)

# Create main directories
parent_path = os.path.abspath(args.out_folder)
train_path = os.path.join(parent_path, 'train')
test_path = os.path.join(parent_path, 'test')
os.makedirs(parent_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

def random_scene(factor):
    global randomizer
    if factor == -1:
        randomizer.random_scene()
    elif factor == 0:
        randomizer.randomize_lighting()
    elif factor == 1:
        randomizer.randomize_geometry()
    else:
        randomizer.randomize_materials()

def scene_json(folder, factor):
    random_scene(factor)
    os.makedirs(folder, exist_ok=True)
    for i in range(settings.views_per_scene):
        randomizer.random_view()
        json_file = folder + "/view%03d.json" % i
        params = randomizer.generate_params()
        randomizer.save_json(json_file, params)

def generate_batch(folder, batch_size, latent_separation):
    randomizer.random_scene()

    if latent_separation:
        os.makedirs(folder, exist_ok=True)
        factor = random.randint(0, 2)
        factor_file = folder + "/factor.txt"
        with open(factor_file, 'w') as fac:
            fac.write(str(factor))
    else:
        factor = -1

    for i in range(batch_size):
        scene_path = os.path.join(folder, "scene%04d" % i)
        scene_json(scene_path, factor)

def generate_set(folder, size, batch_size, latent_separation):
    for i in tqdm(range(size)):
        batch_path = os.path.join(folder, "batch%09d" % i)
        generate_batch(batch_path, batch_size, latent_separation)

print("Generating training data...")
generate_set(train_path, args.size, settings.batch_size, settings.latent_separation)
print("Generating testing data...")
generate_set(test_path, args.testing_size, settings.test_batch_size, False)


    






