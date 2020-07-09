import os
import numpy as np
import torch
import random
import json
from util.settings import *

'''
Functions for reading the config files
'''

def read_json_generator(generator):
    generator_settings = GeneratorSettings()
    if "query_passes" in generator: generator_settings.query_passes = generator["query_passes"]
    if "output_passes" in generator: generator_settings.output_passes = generator["output_passes"]
    if "render_size" in generator: generator_settings.render_size = generator["render_size"]
    if "representations" in generator: generator_settings.representations = generator["representations"]
    if "override_sample" in generator: generator_settings.override_sample = generator["override_sample"]
    if "discard_loss" in generator: generator_settings.discard_loss = generator["discard_loss"]
    if "override_train" in generator: generator_settings.override_train = generator["override_train"]
    if "detach" in generator: generator_settings.detach = generator["detach"]

    if "type" in generator: 
        generator_settings.type = generator["type"]
        if generator["type"] == "Unet":
            generator_settings.settings = UnetSettings()
            if "settings" in generator:
                s = generator["settings"]
                if "loss" in s: generator_settings.settings.loss = s["loss"]
                if "end_relu" in s: generator_settings.settings.end_relu = s["end_relu"]
        elif generator["type"] == "PixelCNN":
            generator_settings.settings = PixelCNNSettings()
            if "settings" in generator:
                s = generator["settings"]
                if "loss" in s: generator_settings.settings.loss = s["loss"]
                if "layers" in s: generator_settings.settings.layers = s["layers"]
                if "hidden_units" in s: generator_settings.settings.hidden_units = s["hidden_units"]
                if "end_relu" in s: generator_settings.settings.end_relu = s["end_relu"]
                if "propagate_buffers" in s: generator_settings.settings.propagate_buffers = s["propagate_buffers"]
                if "propagate_representation" in s: generator_settings.settings.propagate_representation = s["propagate_representation"]
                if "propagate_viewpoint" in s: generator_settings.settings.propagate_viewpoint = s["propagate_viewpoint"]
        elif generator["type"] == "Sum":
            generator_settings.settings = SumSettings()
        elif generator["type"] == "Copy":
            generator_settings.settings = CopySettings()
            if "settings" in generator:
                s = generator["settings"]
                if "index" in s: generator_settings.settings.index = s["index"]
        else:
            generator_settings.settings = GQNSettings()
            if "settings" in generator:
                s = generator["settings"]
                if "cell_type" in s: generator_settings.settings.cell_type = s["cell_type"]
                if "latent_dim" in s: generator_settings.settings.latent_dim = s["latent_dim"]
                if "state_dim" in s: generator_settings.settings.state_dim = s["state_dim"]
                if "core_count" in s: generator_settings.settings.core_count = s["core_count"]
                if "downscaling" in s: generator_settings.settings.downscaling = s["downscaling"]
                if "weight_sharing" in s: generator_settings.settings.weight_sharing = s["weight_sharing"]
                if "upscale_sharing" in s: generator_settings.settings.upscale_sharing = s["upscale_sharing"]
    return generator_settings

def read_json_representation(representation):
    repr_settings = RepresentationSettings()
    if "type" in representation: repr_settings.type = representation["type"]
    if "detach" in representation: repr_settings.detach = representation["detach"]
    if "aggregation_func" in representation: repr_settings.aggregation_func = representation["aggregation_func"]
    if "observation_passes" in representation: repr_settings.observation_passes = representation["observation_passes"]
    if "representation_dim" in representation: repr_settings.representation_dim = representation["representation_dim"]
    if "render_size" in representation: repr_settings.render_size = representation["render_size"]
    
    if "start_sigmoid" in representation: repr_settings.start_sigmoid = representation["start_sigmoid"]
    if "end_sigmoid" in representation: repr_settings.end_sigmoid = representation["end_sigmoid"]
    if "sharpen_sigmoid" in representation: repr_settings.sharp_sigmoid = representation["sharpen_sigmoid"]
    if "sharpening" in representation: repr_settings.sharpening = representation["sharpening"]
    if "final_sharpening" in representation: repr_settings.final_sharpening = representation["final_sharpening"]
    
    if "gradient_reg" in representation: repr_settings.gradient_reg = representation["gradient_reg"]

    return repr_settings

# Reads a JSON config file and sets the settings
def read_json_config(config_file, settings):
    if not os.path.exists(config_file):
        print("Config file: ", config_file, " could not be found!")
        quit()

    with open(config_file, "r") as config:
        data = json.load(config)

    if "logging" in data: settings.logging = data["logging"]
    if "multi_gpu" in data: settings.multi_gpu = data["multi_gpu"]
    if "seed" in data: settings.seed = data["seed"]
    if "checkpoint_interval" in data: settings.checkpoint_interval = data["checkpoint_interval"]
    if "test_interval" in data: settings.test_interval = data["test_interval"]
    if "iterations" in data: settings.iterations = data["iterations"]
    if "batch_size" in data: settings.batch_size = data["batch_size"]
    if "test_batch_size" in data: settings.test_batch_size = data["test_batch_size"]
    if "dataset" in data: settings.dataset = data["dataset"]
    if "cached_dataset" in data: settings.cached_dataset = data["cached_dataset"]
    if "test_data_dir" in data: settings.test_data_dir = data["test_data_dir"]
    if "train_data_dir" in data: settings.train_data_dir = data["train_data_dir"]
    
    if "job_name" in data: settings.job_name = data["job_name"]
    if "job_group" in data: settings.job_group = data["job_group"]
    if "samples_per_pixel" in data: settings.samples_per_pixel = data["samples_per_pixel"]
    if "views_per_scene" in data: settings.views_per_scene = data["views_per_scene"]
    if "random_num_views" in data: settings.random_num_views = data["random_num_views"]
    if "latent_separation" in data: settings.latent_separation = data["latent_separation"]
    if "adaptive_separation" in data: settings.adaptive_separation = data["adaptive_separation"]
    if "empty_partition" in data: settings.empty_partition = data["empty_partition"]
    if "partition_loss" in data: settings.partition_loss = data["partition_loss"]

    if "model" in data:
        model = data["model"]
        if "pose_dim" in model: settings.model.pose_dim = model["pose_dim"]
        if "output_pass" in model: settings.model.output_pass = model["output_pass"]
        
        if "representations" in model:
            representations = model["representations"]
            if len(representations) > 0:
                settings.model.representations = []
                for i in range(len(representations)):
                    settings.model.representations.append(read_json_representation(representations[i]))

        if "generators" in model:
            generators = model["generators"]
            if len(generators) > 0:
                settings.model.generators = []
                for i in range(len(generators)):
                    settings.model.generators.append(read_json_generator(generators[i]))
            
def configure(args, ignore_data=False):
    settings = Settings()
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    parent_dir = os.path.dirname(file_dir)

    if args.config_dir == '':
        args.config_dir = os.path.join(parent_dir, 'configs')
    if not os.path.isdir(args.config_dir):
        print("Config directory: ", args.config_dir, " could not be found.")
        quit()

    # Read a config file that will be used to set up the network
    if args.config != '':
        config_file = os.path.join(args.config_dir, args.config)
        read_json_config(config_file, settings)
        
    dataset_dir = os.path.join(parent_dir, 'datasets', settings.dataset)
    if settings.train_data_dir == '':
        settings.train_data_dir = os.path.join(dataset_dir, 'train')
    if settings.test_data_dir == '':
        settings.test_data_dir = os.path.join(dataset_dir, 'test')
        
    if not ignore_data:
        if not os.path.isdir(settings.train_data_dir) and settings.cached_dataset:
            print("Train directory: ", settings.train_data_dir, " could not be found.")
            quit()
        if not os.path.isdir(settings.test_data_dir) and settings.cached_dataset:
            print("Test directory: ", settings.test_data_dir, " could not be found.")
            quit()

    if settings.seed >= 0:
        seed = settings.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    print("Configuration completed!")
    return settings

def read_checkpoint(args, settings):
    checkpoint = None
    iteration = 1
    if args.checkpoint != '':
        if not os.path.exists(args.checkpoint):
            print("Checkpoint ", args.checkpoint, " could not be found.")
        else:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            iteration = checkpoint['iteration']
            print('Using checkpoint: ' + args.checkpoint)
    return checkpoint, iteration