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
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets
import matplotlib.colors
import copy
from PIL import Image
from tqdm import tqdm

from GQN.model import GenerativeQueryNetwork
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from util.datasets import RTRenderedDataset
from util.config import configure, read_checkpoint
from util.settings import *
from util.attribution import predictions_and_gradients, visualize_buffer_gradients, simple_gradients, representation_gradient

'''
This script visualizes attributions using gradient x input.
'''

parser = ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to load')
parser.add_argument('--config_dir', type=str, default='', help='Where config file is located')
parser.add_argument('--config', type=str, default='', help='Which config to read')
parser.add_argument('--device', type=str, default='', help='Device to run on')
parser.add_argument('--find_checkpoints', action='store_true', help='Attempt to find matching checkpoints automatically')
parser.add_argument('--settings', type=str, default='', help='Load settings from txt file')

parser.add_argument('--out_path', type=str, default='./images/attribution', help='Folder to store the results')
parser.add_argument('--scene_file', type=str, default='', help='Load a given scene file')

args = parser.parse_args()

cuda = torch.cuda.is_available()
if args.device != '':
    device = torch.device(args.device)
else:
    device = torch.device("cuda:0" if cuda else "cpu")

settings = configure(args, ignore_data=True)
checkpoint, iteration = read_checkpoint(args, settings)

settings.batch_size = 1
settings.samples_per_pixel = 1
dataset = RTRenderedDataset(settings, device)
iterator = iter(dataset)
t = 0.0

data = next(iterator)

def format_buffer(buf):
    tmp = buf.clone()
    if tmp.shape[0] == 1:
        tmp = tmp.repeat(3, 1, 1)
    return tmp.detach().cpu().permute(1, 2, 0) ** (1/2.2)

def prepare_data():
    global data
    
    # Tell PyTorch to store gradients wrt input images
    gsize = settings.model.generators[-1].render_size
    for key in data["query_images"]:
        data["query_images"][key] = F.interpolate(data["query_images"][key], size=(gsize, gsize), mode='bilinear').detach()
        data["query_images"][key] = torch.autograd.Variable(data["query_images"][key], requires_grad=True)

    rsize = settings.model.representations[0].render_size
    for key in data["observation_images"]:
        b, n, c, h, w = data["observation_images"][key].shape
        data["observation_images"][key] = data["observation_images"][key].view((b * n, c, h, w))
        data["observation_images"][key] = F.interpolate(data["observation_images"][key], size=(rsize, rsize), mode='bilinear')
        data["observation_images"][key] = data["observation_images"][key].view((b, n, c, rsize, rsize)).detach()
        data["observation_images"][key] = torch.autograd.Variable(data["observation_images"][key], requires_grad=True)

# Create or load a new scene
if args.scene_file == "":
    dataset.renderer.random_scene()
    dataset.renderer.random_view()
else:
    dataset.renderer.load_scene_file(args.scene_file)

# Generate camera view
queries = dataset.get_current_view()
data["query_images"] = queries[0]
data["query_poses"] = queries[1]

# Create observations manually
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

# Prepare data
prepare_data()

iteration = checkpoint['iteration']

# Create network
net = GenerativeQueryNetwork(settings, iteration)
if 'representation_state' in checkpoint and 'generator_state' in checkpoint:
    net.representation.load_state_dict(checkpoint['representation_state'])
    net.generator.load_state_dict(checkpoint['generator_state'])
else:
    net.load_state_dict(checkpoint['model_state'])
net = net.to(device)
net.eval()

print(settings)

# Create matplotlib plots
fig = plt.figure(figsize=(16, 12))
fig.add_subplot(3, 4, 2)
plt.axis('off')
im = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 3)
plt.axis('off')
im2 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 1)
plt.axis('off')
im3 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 5)
plt.axis('off')
im4 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 6)
plt.axis('off')
im5 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 7)
plt.axis('off')
im6 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 9)
plt.axis('off')
im7 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 10)
plt.axis('off')
im8 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
fig.add_subplot(3, 4, 11)
plt.axis('off')
im9 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)

fig.add_subplot(3, 4, 4)
plt.axis('off')
im10 = plt.imshow(format_buffer(data["query_images"][settings.model.output_pass][0]), animated=True)
ax1 = fig.add_subplot(3, 4, 8)
ax2 = fig.add_subplot(3, 4, 12)
ax1.set_ylim([0, 1])
ax2.set_ylim([0, 1])
ax1.set_xlim([0, settings.model.representations[0].representation_dim])
ax2.set_xlim([0, settings.model.representations[0].representation_dim])

# Create plots for representation and its gradients
deltas = 0
representation = net.compute_representations(data)[0][0]
if not settings.latent_separation:
    r_plot, = ax1.plot(representation.cpu().detach().float().numpy().squeeze())
    grad_plot,  = ax2.plot(representation.cpu().detach().float().numpy().squeeze())
else:
    softmax = torch.nn.Softmax(dim=0)
    if settings.multi_gpu:
        deltas = net.representation.representations[0].module.deltas
    else:
        deltas = net.representation.representations[0].deltas
    deltas = settings.model.representations[0].representation_dim * torch.cumsum(softmax(deltas), dim=0)
    deltas = deltas.int()
    light_range = range(0, deltas[0])
    geometry_range = range(deltas[0], deltas[1])
    material_range = range(deltas[1], settings.model.representations[0].representation_dim)
    light_plot, = ax1.plot(light_range, representation[0][:deltas[0]].cpu().detach().float().numpy().squeeze(), 'r')
    geom_plot, = ax1.plot(geometry_range, representation[0][deltas[0]:deltas[1]].cpu().detach().float().numpy().squeeze(), 'g')
    mat_plot,  = ax1.plot(material_range, representation[0][deltas[1]:].cpu().detach().float().numpy().squeeze(), 'b')
    
    light_grad_plot, = ax2.plot(light_range, representation[0][:deltas[0]].cpu().detach().float().numpy().squeeze(), 'r')
    geom_grad_plot, = ax2.plot(geometry_range, representation[0][deltas[0]:deltas[1]].cpu().detach().float().numpy().squeeze(), 'g')
    mat_grad_plot,  = ax2.plot(material_range, representation[0][deltas[1]:].cpu().detach().float().numpy().squeeze(), 'b')

ax1.title.set_text('Representation')
ax2.title.set_text('Gradient')

# Function for computing attributions based on settings
def compute_attributions(data, options, pixel, patch_size):
    global net

    # Evaluate settings
    mode = 'pixel'
    
    if options[2] and options[1]:
        mode = 'partition'
    elif options[1]:
        mode = 'representation'
    elif options[0]:
        mode = 'mean'

    imsize = settings.model.generators[-1].render_size
    px = [int(pixel[0] * imsize), int(pixel[1] * imsize)]
    patch = [[px[0], px[0] + int(patch_size)], [px[1], px[1] + int(patch_size)]]
    

    if mode == 'partition' or mode == 'representation':
        # Representation attribution
        if settings.multi_gpu:
            repr_net = net.representation.representations[0].module
        else:
            repr_net = net.representation.representations[0]
        r_dim = settings.model.representations[0].representation_dim

        if mode == 'partition':
            p = min(2, int(pixel[0] * 3))
            softmax = torch.nn.Softmax(dim=0)
            deltas = repr_net.deltas
            deltas = r_dim * torch.cumsum(softmax(deltas), dim=0)
            deltas = deltas.int()
            if p == 0:
                part = [0, deltas[0]]
            elif p == 1:
                part = [deltas[0], deltas[1]]
            else:
                part = [deltas[1], deltas[2]]
        else:
            part = int(pixel[0] * (r_dim - 1))
            part = [part, part+1]

        predictions, query_grads, observation_grads, representation_grads = representation_gradient(data, net, settings, part=part)
    else:
        # Patch attribution
        if mode == 'mean':
            patch = []
        predictions, query_grads, observation_grads, representation_grads = simple_gradients(data, net, settings, patch=patch)
 
    return predictions, observation_grads, query_grads, representation_grads

query_grads = 0
observation_grads = 0
representation_grads = 0
predictions = 0

# This function updates the plots and calls for new attributions to be computed
def update(val):
    global data, net, settings, query_grads, observation_grads, representation_grads, predictions
    
    # Get current settings from the different checkboxes and sliders
    pixelx = (data["query_images"][settings.model.output_pass].shape[3] - 1) * spixelx.val
    pixely = (data["query_images"][settings.model.output_pass].shape[2] - 1) * spixely.val
    pixelx = int(pixelx)
    pixely = int(pixely)
    button_status = bmode.get_status()
    selected = observation_buttons.value_selected
    selected_query = query_buttons.value_selected
    minimum = smin_vis.val
    maximum = smax_vis.val
    patch_size = spatch.val

    # Compute predictions and gradients
    predictions, observation_grads, query_grads, representation_grads = compute_attributions(data, button_status, [spixelx.val, spixely.val], patch_size)

    with torch.no_grad():
        visualize_result = format_buffer(predictions[settings.model.output_pass][0])
        if not button_status[0] and not button_status[1]:
            red = torch.Tensor([1, 0, 0])
            patch_y = min(pixely + int(patch_size), int(visualize_result.shape[0]))
            patch_x = min(pixelx + int(patch_size), int(visualize_result.shape[1]))
            visualize_result[pixely:patch_y, pixelx:patch_x, :] = red
        im3.set_data(format_buffer(data["query_images"]["beauty"][0]))
        im2.set_data(visualize_result)
        im.set_data(format_buffer(data["query_images"][settings.model.output_pass][0]))

        # Visualize beauty images
        im4.set_data(format_buffer(data["observation_images"][selected][0][0]))
        im5.set_data(format_buffer(data["observation_images"][selected][0][1]))
        im6.set_data(format_buffer(data["observation_images"][selected][0][2]))

        # Save magnitudes for representation
        repr_mag.set_text(str(representation_grads.sum().item())[:9])
        if settings.latent_separation:
            light_mag.set_text(str(representation_grads[:, :deltas[0]].sum().item())[:9])
            geo_mag.set_text(str(representation_grads[:, deltas[0]:deltas[1]].sum().item())[:9])
            mat_mag.set_text(str(representation_grads[:, deltas[1]:].sum().item())[:9])
            
        # Save magnitudes for observations
        for key in observation_magnitudes:
            if key in observation_grads:
                grad = torch.sqrt(torch.sum(observation_grads[key] * observation_grads[key], dim=2, keepdim=True))
                observation_magnitudes[key][0].set_text(str(grad[0][0].sum().item())[:9])
                observation_magnitudes[key][1].set_text(str(grad[0][1].sum().item())[:9])
                observation_magnitudes[key][2].set_text(str(grad[0][2].sum().item())[:9])

        # Save magnitudes for queries
        for key in query_magnitudes:
            if key in query_grads:
                grad = torch.sqrt(torch.sum(query_grads[key] * query_grads[key], dim=1, keepdim=True))
                query_magnitudes[key].set_text(str(grad.sum().item())[:9])        

        # Get gradient visualization
        b, n, c, h, w = observation_grads[selected].shape
        obs_gradients = observation_grads[selected].view((b * n, c, h, w))
        obs_gradients = visualize_buffer_gradients(obs_gradients, heatmap=button_status[3], maximum=maximum, minimum=minimum)
        obs_gradients = obs_gradients.view((b, n, 3, h, w))

        # Modify visualization of keys
        obs_visualize = data["observation_images"][selected]
        if selected == "normal":
            obs_visualize = (obs_visualize + 1) * 0.5
        elif selected == "position":
            obs_visualize = (obs_visualize + 3) * (1/6)
        obs_visualize = obs_visualize.clamp(0, 1)

        # Heatmap is overlayed while gradients are otherwise multiplied
        if button_status[3]:
            obs_gradients = 0.5 * obs_gradients + 0.5 * obs_visualize

        im7.set_data(format_buffer(obs_gradients[0][0]))
        im8.set_data(format_buffer(obs_gradients[0][1]))
        im9.set_data(format_buffer(obs_gradients[0][2]))
                
        # Visualize gradients for selected query buffer
        if selected_query in query_grads:
            query_gradients = query_grads[selected_query]
            query_gradients = visualize_buffer_gradients(query_gradients, heatmap=button_status[3], maximum=maximum, minimum=minimum)

            query_visualize = data["query_images"][selected_query]
            if selected_query == "normal":
                query_visualize = (query_visualize + 1) * 0.5
            elif selected_query == "position":
                query_visualize = (query_visualize + 3) * (1/6)
            query_visualize = query_visualize.clamp(0, 1)

            if button_status[3]:
                query_gradients = 0.5 * query_gradients + 0.5 * query_visualize
        else:
            query_gradients = torch.zeros_like(obs_gradients[0])
        im10.set_data(format_buffer(query_gradients[0]))
        
        if settings.multi_gpu:
            repr_net = net.representation.representations[0].module
        else:
            repr_net = net.representation.representations[0]
        representation = repr_net.representation[0]
        # Normalize
        representation /= representation.max()
        representation_grads_normed = representation_grads / representation_grads.max()

        # Plot representation
        if settings.latent_separation:
            light_plot.set_ydata(representation[:deltas[0]].cpu().detach().float().numpy().squeeze())
            geom_plot.set_ydata(representation[deltas[0]:deltas[1]].cpu().detach().float().numpy().squeeze())
            mat_plot.set_ydata(representation[deltas[1]:].cpu().detach().float().numpy().squeeze())

            light_grad_plot.set_ydata(representation_grads_normed.mean(dim=0)[:deltas[0]].cpu().detach().float().numpy().squeeze())
            geom_grad_plot.set_ydata(representation_grads_normed.mean(dim=0)[deltas[0]:deltas[1]].cpu().detach().float().numpy().squeeze())
            mat_grad_plot.set_ydata(representation_grads_normed.mean(dim=0)[deltas[1]:].cpu().detach().float().numpy().squeeze())
        else:
            r_plot.set_ydata(representation.cpu().detach().float().numpy().squeeze())
            grad_plot.set_ydata(representation_grads_normed.mean(dim=0).cpu().detach().float().numpy().squeeze())

    fig.canvas.draw()
    fig.canvas.flush_events()

# This loads a saved settings .txt file
def load(file_path):
    f = open(args.settings, 'r')
    lines = f.readlines()
    button_status = lines[0].strip().split(" ")
    button_status = [bool(int(x)) for x in button_status]
    for i in range(len(button_status)):
        if button_status[i] != bmode.get_status()[i]:
            bmode.set_active(i)

    minimum = float(lines[1].strip())
    maximum = float(lines[2].strip())

    pixelx = float(lines[3].strip())
    pixely = float(lines[4].strip())

    selected = lines[5].strip()
    selected_query = lines[6].strip()

    patch_size = float(lines[7].strip())

    scene_file = lines[8].strip()

    smin_vis.set_val(minimum)
    smax_vis.set_val(maximum)
    spixelx.set_val(pixelx)
    spixely.set_val(pixely)
    observation_buttons.value_selected = selected
    query_buttons.value_selected = selected_query

    spatch.set_val(patch_size)

    f.close()

    # Load scene
    dataset.renderer.load_scene_file(scene_file)


# Saves settings and all attribution values and a shell command to reproduce results
def save(val):
    global data, net, settings, query_grads, observation_grads, representation_grads, predictions, deltas
    button_status = bmode.get_status()
    minimum = smin_vis.val
    maximum = smax_vis.val
    pixelx = spixelx.val
    pixely = spixely.val
    selected = observation_buttons.value_selected
    selected_query = query_buttons.value_selected
    patch_size = spatch.val

    # Save settings
    path = os.path.join(args.out_path, str(time.time()))
    print('Saving data in %s ...' %(path))
    os.makedirs(path, exist_ok=True)
    button_stat = " ".join([str(int(x)) for x in button_status])
    f = open(path + "/settings.txt", 'w')
    f.write(button_stat + "\n")
    f.write(str(minimum) + "\n")
    f.write(str(maximum) + "\n")
    f.write(str(pixelx) + "\n")
    f.write(str(pixely) + "\n")
    f.write(selected + "\n")
    f.write(selected_query + "\n")
    f.write(str(patch_size) + "\n")
    json_path = path + "/scene.json"
    f.write(json_path + "\n")
    f.close()

    # Save JSON scene file
    params = dataset.renderer.randomizer.generate_params()
    dataset.renderer.randomizer.save_json(json_path, params)

    # Save fig
    plt.savefig(os.path.join(path) + "/figure.png")

    # Save query images
    query_path = os.path.join(path, "generator")
    os.makedirs(query_path, exist_ok=True)
    for key in settings.model.generators[-1].query_passes:
        X = data["query_images"][key]
        np.savez_compressed(query_path + "/" + key + ".npz", X.detach().cpu().numpy())
        X = F.interpolate(X, size=(4 * X.shape[2], 4 * X.shape[3]), mode='nearest')
        X = format_buffer(X[0]).clamp(0, 1).numpy()

        if key in query_grads:
            dX = query_grads[key]
            np.savez_compressed(query_path + "/" + key + "_grad.npz", dX.detach().cpu().numpy())
            dX = visualize_buffer_gradients(dX, heatmap=button_status[3], maximum=maximum, minimum=minimum)
        else:
            dX = torch.zeros_like(data["query_images"][key])    
        dX = F.interpolate(dX, size=(4 * dX.shape[2], 4 * dX.shape[3]), mode='nearest')
        dX = format_buffer(dX[0]).clamp(0, 1).numpy()
        
        Xu8 = Image.fromarray((X * 255).astype(np.uint8))
        Xu8.save(query_path + "/" + key + ".jpg", quality=100)

        dXu8 = Image.fromarray((dX * 255).astype(np.uint8))
        dXu8.save(query_path + "/" + key + "_grad.jpg", quality=100)

    # Save predictions
    for key in settings.model.generators[-1].output_passes:
        R = data["query_images"][key]
        np.savez_compressed(query_path + "/" + key + "_ref.npz", R.detach().cpu().numpy())
        R = F.interpolate(R, size=(4 * R.shape[2], 4 * R.shape[3]), mode='nearest')
        R = format_buffer(R[0]).clamp(0, 1).numpy()

        G = predictions[key]
        np.savez_compressed(query_path + "/" + key + "_gen.npz", G.detach().cpu().numpy())
        G = F.interpolate(G, size=(4 * G.shape[2], 4 * G.shape[3]), mode='nearest')
        G = format_buffer(G[0]).clamp(0, 1).numpy()

        Ru8 = Image.fromarray((R * 255).astype(np.uint8))
        Ru8.save(query_path + "/" + key + "_ref.jpg", quality=100)

        Gu8 = Image.fromarray((G * 255).astype(np.uint8))
        Gu8.save(query_path + "/" + key + "_gen.jpg", quality=100)

    # Save observation images
    observation_path = os.path.join(path, "representation")
    os.makedirs(observation_path, exist_ok=True)
    for key in settings.model.representations[0].observation_passes:
        O = data["observation_images"][key]
        for n in range(O.shape[1]):
            X = O[0, n].unsqueeze(0)
            np.savez_compressed(observation_path + "/" + key + str(n) + ".npz", X.detach().cpu().numpy())
            X = F.interpolate(X, size=(4 * X.shape[2], 4 * X.shape[3]), mode='nearest')
            X = format_buffer(X[0]).clamp(0, 1).numpy()

            dX = observation_grads[key][0][n].unsqueeze(0)
            np.savez_compressed(observation_path + "/" + key + str(n) + "_grad.npz", dX.detach().cpu().numpy())
            dX = visualize_buffer_gradients(dX, heatmap=button_status[3], maximum=maximum, minimum=minimum)

            dX = F.interpolate(dX, size=(4 * dX.shape[2], 4 * dX.shape[3]), mode='nearest')
            dX = format_buffer(dX[0]).clamp(0, 1).numpy()

            Xu8 = Image.fromarray((X * 255).astype(np.uint8))
            Xu8.save(observation_path + "/" + key + str(n) + ".jpg", quality=100)

            dXu8 = Image.fromarray((dX * 255).astype(np.uint8))
            dXu8.save(observation_path + "/" + key + str(n) + "_grad.jpg", quality=100)

    # Save .npz of representation
    if settings.multi_gpu:
        repr_net = net.representation.representations[0].module
    else:
        repr_net = net.representation.representations[0]
    r = repr_net.representation.detach().cpu().float().numpy().squeeze()
    rgrad = representation_grads.cpu().detach().float().numpy().squeeze()
    geometry = [-1, -1]
    lighting = [-1, -1]
    materials = [-1, -1]
    if settings.latent_separation and settings.adaptive_separation:
        lighting = [0, int(deltas[0])]
        geometry = [int(deltas[0]), int(deltas[1])]
        materials = [int(deltas[1]), int(deltas[2])]
    print(lighting, geometry, materials)
    np.savez_compressed(observation_path + "/R.npz", representation=r, gradient=rgrad, lighting=lighting, geometry=geometry, materials=materials)



def generate_scene(val):
    global data
    
    # Random scene and render query view
    dataset.renderer.random_scene()
    dataset.renderer.random_view()
    queries = dataset.get_current_view()
    data["query_images"] = queries[0]
    data["query_poses"] = queries[1]

    # Create observations manually
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
    prepare_data()    
    update(0)

# Function that zeroes out material partition
def replace_mat(val):
    global data
    representations = net.compute_representations(data)[0]
    representations[0][:, deltas[1]:] = 0
    sample_image = net.sample(data, representations=representations)
    visualize_result = format_buffer(sample_image[-1][settings.model.output_pass][0])
    im2.set_data(visualize_result)

# Function that zeroes out geometry partition
def replace_geo(val):
    global data
    representations = net.compute_representations(data)[0]
    representations[0][:, deltas[0]:deltas[1]] = 0
    sample_image = net.sample(data, representations=representations)
    visualize_result = format_buffer(sample_image[-1][settings.model.output_pass][0])
    im2.set_data(visualize_result)

# Function that zeroes out lighting partition
def replace_light(val):
    global data
    representations = net.compute_representations(data)[0]
    representations[0][:, :deltas[0]] = 0
    sample_image = net.sample(data, representations=representations)
    visualize_result = format_buffer(sample_image[-1][settings.model.output_pass][0])
    im2.set_data(visualize_result)

def replace_all(val):
    global data
    representations = net.compute_representations(data)[0]
    representations[0][:, :, :, :] = 0
    sample_image = net.sample(data, representations=representations)
    visualize_result = format_buffer(sample_image[-1][settings.model.output_pass][0])
    im2.set_data(visualize_result)

# 
# This section sets up the matplotlib window with buttons and settings
#

# Sliders
axcolor = 'lightgoldenrodyellow'
axpixelx = plt.axes([0.25, 0.015, 0.5, 0.01], facecolor=axcolor)
axpixely = plt.axes([0.25, 0.035, 0.5, 0.01], facecolor=axcolor)
axmin_vis = plt.axes([0.25, 0.055, 0.5, 0.01], facecolor=axcolor)
axmax_vis = plt.axes([0.25, 0.075, 0.5, 0.01], facecolor=axcolor)
axpatch = plt.axes([0.25, 0.095, 0.5, 0.01], facecolor=axcolor)

spixelx = matplotlib.widgets.Slider(axpixelx, 'Pixel X', 0.0, 1.0, valinit=0.5, valstep=0.0001)
spixely = matplotlib.widgets.Slider(axpixely, 'Pixel Y', 0.0, 1.0, valinit=0.5, valstep=0.0001)
smin_vis = matplotlib.widgets.Slider(axmin_vis, 'Range Min', 0.0, 0.005, valinit=0.0, valstep=0.000001)
smax_vis = matplotlib.widgets.Slider(axmax_vis, 'Range Max', 0.0, 5, valinit=0.00025, valstep=0.000001)
spatch = matplotlib.widgets.Slider(axpatch, 'Patch Size', 1, 64, valinit=1, valstep=1)


# Buttons
axmode = plt.axes([0.1295, 0.9125, 0.125, 0.065], facecolor='lightblue')
bmode = matplotlib.widgets.CheckButtons(axmode, ["Activate Mean", "Representation Grad", "Partition Mean", "Heatmap"], actives=[False, False, False, True])

axscene = plt.axes([0.2401, 0.9125, 0.1, 0.065], facecolor='lightblue')
bscene = matplotlib.widgets.Button(axscene, 'New Scene')

axsave = plt.axes([0.3402, 0.9125, 0.1, 0.065], facecolor='lightblue')
bsave = matplotlib.widgets.Button(axsave, 'Save')

axreplace_light = plt.axes([0.451, 0.9125, 0.1, 0.065], facecolor='lightcoral')
breplace_light = matplotlib.widgets.Button(axreplace_light, 'Replace Light')

axreplace_geo = plt.axes([0.551, 0.9125, 0.1, 0.065], facecolor='lightgreen')
breplace_geo = matplotlib.widgets.Button(axreplace_geo, 'Replace Geo')

axreplace_mat = plt.axes([0.651, 0.9125, 0.1, 0.065], facecolor='lightblue')
breplace_mat = matplotlib.widgets.Button(axreplace_mat, 'Replace Mat')

axreplace_all = plt.axes([0.751, 0.9125, 0.1, 0.065], facecolor='lightgray')
breplace_all = matplotlib.widgets.Button(axreplace_all, 'Replace All')

# Text
test = plt.axes([0.9125, 0.05, 0.075, 0.9], facecolor='lightgray', xticks=[], yticks=[])
plt.text(0.1, 0.98, "Magnitudes:", fontweight='bold')
plt.text(0.1, 0.96, "Representation:", fontweight='medium')
repr_mag = plt.text(0.1, 0.94, "0.0")

plt.text(0.1, 0.92, "Lighting:", fontweight='medium')
light_mag = plt.text(0.1, 0.90, "0.0")
plt.text(0.1, 0.88, "Geometry:", fontweight='medium')
geo_mag = plt.text(0.1, 0.86, "0.0")
plt.text(0.1, 0.84, "Material:", fontweight='medium')
mat_mag = plt.text(0.1, 0.82, "0.0")

plt.text(0.1, 0.80, "Observations", fontweight='bold')
observation_magnitudes = {}
pointer = 0.78
for buffer in settings.model.representations[0].observation_passes:
    observation_magnitudes[buffer] = []
    plt.text(0.1, pointer, buffer, fontweight='medium')
    pointer -= 0.02
    observation_magnitudes[buffer].append(plt.text(0.1, pointer, "0.0"))
    pointer -= 0.02
    observation_magnitudes[buffer].append(plt.text(0.1, pointer, "0.0"))
    pointer -= 0.02
    observation_magnitudes[buffer].append(plt.text(0.1, pointer, "0.0"))
    pointer -= 0.02

plt.text(0.1, pointer, "Query Buffers", fontweight='bold')
pointer -= 0.02
query_magnitudes = {}
for buffer in settings.model.generators[-1].query_passes:
    plt.text(0.1, pointer, buffer, fontweight='medium')
    pointer -= 0.02
    query_magnitudes[buffer] = plt.text(0.1, pointer, "0.0")
    pointer -= 0.02

# Buffer Selection
observation_buffers = settings.model.representations[0].observation_passes
num_observation_buffers = len(observation_buffers)
axbuffers = plt.axes([0.0125, 0.3 - (0.025 * num_observation_buffers), 0.1, 0.05 * num_observation_buffers], facecolor=axcolor)
observation_buttons = matplotlib.widgets.RadioButtons(axbuffers, labels=observation_buffers)

query_buffers = settings.model.generators[-1].query_passes
num_query_buffers = len(query_buffers)
axquerybuffers = plt.axes([0.0125, 0.825 - (0.025 * num_query_buffers), 0.1, 0.05 * num_query_buffers], facecolor=axcolor)
query_buttons = matplotlib.widgets.RadioButtons(axquerybuffers, labels=query_buffers, active=1)

if args.settings != '':
    load(args.settings)

# Hooks
observation_buttons.on_clicked(update)
query_buttons.on_clicked(update)
breplace_all.on_clicked(replace_all)
breplace_mat.on_clicked(replace_mat)
bmode.on_clicked(update)
bscene.on_clicked(generate_scene)
bsave.on_clicked(save)
breplace_light.on_clicked(replace_light)
breplace_geo.on_clicked(replace_geo)
spixelx.on_changed(update)
spixely.on_changed(update)
smin_vis.on_changed(update)
smax_vis.on_changed(update)
spatch.on_changed(update)

update(0.)

fig.canvas.set_window_title('Visualize Attributions')

plt.show()
