import numpy as np
import os
import math
import random
import time
import torch

from argparse import ArgumentParser
from util.config import configure
from renderer.optix_renderer.render import OptixRenderer
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--config_dir', type=str, default='', help='Where config file is located')
parser.add_argument('--config', type=str, default='', help='Config file to read')
parser.add_argument('--device', type=int, default=0, help='Which device to run on')
parser.add_argument('--in_folder', type=str, default='', help='What folder to run through')
parser.add_argument('--out_folder', type=str, default='', help='Folder to output batches to')
parser.add_argument('--find_checkpoints', action='store_true', help='Attempt to find checkpoints automatically')
args = parser.parse_args()

if args.in_folder == '':
    print("Sorry, no in_folder provided. Quitting...")
    quit()

if args.out_folder == '':
    print("Sorry, no out_folder provided. Quitting...")
    quit()

settings = configure(args, ignore_data=True)
render_size = 0
for g in settings.model.generators:
    render_size = max(render_size, g.render_size)
for e in settings.model.representations:
    render_size = max(render_size, e.render_size)

renderer = OptixRenderer(render_size, args.device)

def render_json(json_path):
    global renderer
    renderer.load_scene_file(json_path)
    passes = renderer.draw_scene(settings.samples_per_pixel)
    poses = renderer.get_pose()
    return passes, poses

# Render a folder containing multiple json files of the same scene
# These passes and poses are then concatenated to create new observations
def render_scene(folder):
    files = sorted(os.listdir(folder))
    num_views = 0
    view_passes = torch.zeros([1])
    view_poses = torch.zeros([1])

    for relf in files:
        f = folder + "/" + relf
        if os.path.isfile(f) and os.path.splitext(f)[1] == ".json":
            passes, pose = render_json(f)
            print("FILE:", f)

            for p in passes:
                passes[p] = torch.from_numpy(passes[p]).float().cpu().permute(2, 0, 1)
                passes[p] = passes[p].reshape((1, 1, *passes[p].shape))

            pose = torch.from_numpy(pose).float().cpu()
            pose = pose.reshape(1, 1, pose.shape[0], 1, 1)

            # Concatenate to previous render results
            if num_views == 0:
                view_passes = passes
                view_poses = pose
            else:
                for key in passes:
                    view_passes[key] = torch.cat([view_passes[key], passes[key]], dim=1)
                view_poses = torch.cat([view_poses, pose], dim=1)
            num_views += 1

    if num_views == 0:
        print("NO JSON FILES FOUND IN FOLDER:", folder)

    return view_passes, view_poses

def render_and_save_batch(folder):
    folders = sorted(os.listdir(folder))
    num_scenes = 0
    batch_passes = torch.zeros([1])
    batch_poses = torch.zeros([1])

    # Run through every folder and render folders containing JSON files
    for relfolder in folders:
        absfolder = os.path.join(folder, relfolder)
        if os.path.isdir(absfolder):
            passes, poses = render_scene(absfolder)

            if num_scenes == 0:
                batch_passes = passes
                batch_poses = poses
            else:
                for key in passes:
                    batch_passes[key] = torch.cat([batch_passes[key], passes[key]], dim=0)
                batch_poses = torch.cat([batch_poses, poses], dim=0)

            num_scenes += 1

    batch_poses = batch_poses.numpy().astype('float16')
    for key in batch_passes:
        batch_passes[key] = batch_passes[key].numpy().astype('float16')

    if num_scenes == 0:
        print("NO BATCHES IN FOLDER:", folder)
    else:
        # Read factor file
        factor_file = folder + "/factor.txt"
        if os.path.exists(factor_file):
            fac = open(factor_file, 'r')
            factor = int(fac.readlines()[0])
            fac.close()
        else:
            factor = -1

        # Save a .npz file containing all the rendered scenes
        batch_name = folder.split("/")[-1] + str(time.time()) + ".npz"
        batch_file = args.out_folder + "/" + batch_name
        np.savez_compressed(batch_file, **batch_passes, poses=batch_poses, factor=factor)

def process_root_folder(folder):
    batch_folders = sorted(os.listdir(folder))
    for i in tqdm(range(len(batch_folders))):
        batch_folder = os.path.join(folder, batch_folders[i])
        if os.path.isdir(batch_folder):
            print("Rendering folder:", batch_folder)
            render_and_save_batch(batch_folder)
        
# PROCESSING FOLDER
os.makedirs(args.out_folder, exist_ok=True)
process_root_folder(args.in_folder)




    



