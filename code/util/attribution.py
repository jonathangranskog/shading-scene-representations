import torch
import torch.nn as nn
import copy
import time
import tqdm

def fetch_gradients(data, settings):
    query_grads = {}
    observation_grads = {}
    for key in data["query_images"]:
        if data["query_images"][key].grad is not None:
            query_grads[key] = data["query_images"][key].grad * data["query_images"][key]
    for key in data["observation_images"]:
        if data["observation_images"][key].grad is not None:
            observation_grads[key] = data["observation_images"][key].grad * data["observation_images"][key]

    return query_grads, observation_grads

# Function that computes the gradients for a single channel in a pixel
def pixel_channel_gradient(data, model, settings, image, pixel, channel):
    model.zero_grad()
    value = image[0, channel, pixel[1], pixel[0]]
    model.representation.retain_gradients()
    value.backward(retain_graph=True)

    # Assume we only have one representation!
    if settings.multi_gpu:
        repr_net = model.representation.representations[0].module
    else:
        repr_net = model.representation.representations[0]

    rgrad = repr_net.representation.grad * repr_net.representation
    qgrad, ograd = fetch_gradients(data, settings)
    return qgrad, ograd, rgrad

def pixel_gradient(data, model, settings, image, pixel):
    query_grads = {}
    observation_grads = {}

    assert(image.shape[0] == 1)

    # Compute gradient matrices with shape [num_channels_output, num_channels_input, height, width]
    for channel in range(image.shape[1]):
        qgrad, ograd, rgrad = pixel_channel_gradient(data, model, settings, image, pixel, channel)
        if channel == 0:
            query_grads = qgrad
            observation_grads = ograd
            representation_grads = rgrad
        else:
            for key in qgrad:
                query_grads[key] = torch.cat([query_grads[key], qgrad[key]], dim=0)
            for key in ograd:
                observation_grads[key] = torch.cat([observation_grads[key], ograd[key]], dim=0)
            representation_grads = torch.cat([representation_grads, rgrad], dim=0)
    
    return query_grads, observation_grads, representation_grads

def mean_patch_gradient(data, model, settings, image, patch_bounds, signed=False):
    query_grad = {}
    observation_grad = {} 
    representation_grad = []

    patch_size = 0

    for x in tqdm.tqdm(range(patch_bounds[0][0], patch_bounds[0][1])):
        for y in range(patch_bounds[1][0], patch_bounds[1][1]):
            qgrad, ograd, rgrad = pixel_gradient(data, model, settings, image, [x, y])
            if not signed:
                for key in qgrad:
                    qgrad[key] = torch.abs(qgrad[key])
                for key in ograd:
                    ograd[key] = torch.abs(ograd[key])
                rgrad = torch.abs(rgrad)

            if patch_size == 0:
                query_grad = qgrad
                observation_grad = ograd
                representation_grad = rgrad
            else:
                representation_grad += rgrad
                for key in query_grad:
                    query_grad[key] += qgrad[key]
                for key in observation_grad:
                    observation_grad[key] += ograd[key]
            patch_size += 1

    representation_grad = representation_grad  / patch_size
    for key in query_grad:
        query_grad[key] = torch.mean(query_grad[key], dim=0, keepdim=True) / patch_size
    for key in observation_grad:
        observation_grad[key] = torch.mean(observation_grad[key], dim=0, keepdim=True)  / patch_size
    return query_grad, observation_grad, representation_grad

def representation_gradient(data, model, settings, part=[]):
    
    batch_data = copy.deepcopy(data)
    for key in data["query_images"]:
        batch_data["query_images"][key] = torch.autograd.Variable(batch_data["query_images"][key], requires_grad=True)
    for key in data["observation_images"]:
        batch_data["observation_images"][key] = torch.autograd.Variable(batch_data["observation_images"][key], requires_grad=True)

    sample_image = model.sample(batch_data)
    if settings.multi_gpu:
        repr_net = model.representation.representations[0].module
    else:
        repr_net = model.representation.representations[0]
    representation = repr_net.representation

    assert(representation.shape[0] == 1)

    for dim in range(part[0], part[1]):
        model.zero_grad()
        value = representation[0, dim]
        model.representation.retain_gradients()
        value.backward(retain_graph=True)

        _, ograd = fetch_gradients(batch_data, settings)

        # Compute gradient matrices with shape [r_dim, num_channels_input, height, width]
        if dim == part[0]:
            observation_grads = ograd
        else:
            for key in ograd:
                observation_grads[key] = torch.cat([observation_grads[key], ograd[key]], dim=0)

    for key in observation_grads:
        observation_grads[key] = torch.sum(observation_grads[key], dim=0, keepdim=True)

    query_grads = {}
    for key in settings.model.representations[0].observation_passes:
        if batch_data["observation_images"][key].grad is not None:
            observation_grads[key] = batch_data["observation_images"][key].grad * batch_data["observation_images"][key]

    if repr_net.representation.grad is not None:
        representation_grads = repr_net.representation.grad
    else:
        representation_grads = torch.zeros_like(repr_net.representation)

    return sample_image[-1], query_grads, observation_grads, representation_grads


def simple_gradients(data, model, settings, patch=[]):
    t0 = time.time()
    print("Computing gradients...")
    batch_data = copy.deepcopy(data)
    for key in data["query_images"]:
        batch_data["query_images"][key] = torch.autograd.Variable(batch_data["query_images"][key], requires_grad=True)
    for key in data["observation_images"]:
        batch_data["observation_images"][key] = torch.autograd.Variable(batch_data["observation_images"][key], requires_grad=True)

    sample_image = model.sample(batch_data)

    imsize = settings.model.generators[-1].render_size
    if len(patch) == 0:
        patch = [[0, imsize], [0, imsize]]

    qgrad, ograd, rgrad = mean_patch_gradient(batch_data, model, settings, sample_image[-1]["beauty"], patch, signed=False)

    t1 = time.time()
    print(str(t1 - t0))

    return sample_image[-1], qgrad, ograd, rgrad

# Function that takes some data as input and computes the gradient wrt the input
# The gradient can be computed for the representation or the output image
def predictions_and_gradients(data, model, settings, mode='mean', pixel=[0.5, 0.5], patch_size=1, partition=0, r_index=0, g_index=-1):
    # Prepare data by telling PyTorch to store gradients for input data
    batch_data = copy.deepcopy(data)
    
    for key in data["query_images"]:
        batch_data["query_images"][key] = torch.autograd.Variable(batch_data["query_images"][key], requires_grad=True)
    for key in data["observation_images"]:
        batch_data["observation_images"][key] = torch.autograd.Variable(batch_data["observation_images"][key], requires_grad=True)
    
    # Get predictions
    model.zero_grad()
    sample_image = model.sample(batch_data)

    if settings.multi_gpu:
        repr_net = model.representation.representations[r_index].module
    else:
        repr_net = model.representation.representations[r_index]

    # Select what to compute gradient for
    if mode == 'mean' or mode == 'pixel':
        outputs = list(sample_image[g_index].values())
        output_image = outputs[0]
        for i in range(1, len(outputs)):
            output_image = torch.cat([output_image, outputs[i]], dim=1)
        
        if mode == 'mean':
            prop = output_image.mean()
        else:
            pixely0 = int((output_image.shape[2] - 1) * pixel[1])
            pixelx0 = int((output_image.shape[3] - 1) * pixel[0])
            pixely1 = min(pixely0 + int(patch_size), output_image.shape[2])
            pixelx1 = min(pixelx0 + int(patch_size), output_image.shape[3])
            prop = output_image[:, :, pixely0:pixely1, pixelx0:pixelx1].mean()
    else:
        r = model.compute_representations(batch_data)[0][r_index]

        # Partition specific gradients
        if settings.latent_separation and mode == 'partition':
            softmax = torch.nn.Softmax(dim=0)
            deltas = repr_net.deltas
            deltas = r.shape[1] * torch.cumsum(softmax(deltas), dim=0)
            deltas = deltas.int()

            if partition == 0:
                prop = r[:, :deltas[0]].mean()
            elif partition == 1:
                prop = r[:, deltas[0]:deltas[1]].mean()
            else:
                prop = r[:, deltas[1]:].mean()
        else:
            # Gradient wrt single element in representation
            prop = r[:, int(pixel[0] * (r.shape[1] - 1))].mean()
    
    # Run backward pass
    model.representation.retain_gradients()
    prop.backward()

    if repr_net.representation.grad is not None:
        representation_grads = repr_net.representation.grad * repr_net.representation
    else:
        representation_grads = torch.zeros_like(repr_net.representation)
    query_grads = {}
    observation_grads = {}
    if mode == 'mean' or mode == 'pixel':
        for key in settings.model.generators[g_index].query_passes:
            if batch_data["query_images"][key].grad is not None:
                query_grads[key] = batch_data["query_images"][key].grad * batch_data["query_images"][key]
    for key in settings.model.representations[r_index].observation_passes:
        if batch_data["observation_images"][key].grad is not None:
            observation_grads[key] = batch_data["observation_images"][key].grad * batch_data["observation_images"][key]

    return sample_image[g_index], observation_grads, query_grads, representation_grads

# Convert a whole hsv[0,1] tensor to rgb
def hsv_to_rgb(tensor):
    C = tensor[:, 2:] * tensor[:, 1:2]
    X = C * (1 - torch.abs(torch.fmod(tensor[:, 0:1] * 6, 2) - 1))
    m = tensor[:, 2:] - C

    H = tensor[:, 0:1] * 360
    
    zeros = torch.zeros_like(C)
    rgb = torch.cat([C, zeros, X], dim=1)
    rgb = torch.where(H < 300, torch.cat([X, zeros, C], dim=1), rgb)
    rgb = torch.where(H < 240, torch.cat([zeros, X, C], dim=1), rgb)
    rgb = torch.where(H < 180, torch.cat([zeros, C, X], dim=1), rgb)
    rgb = torch.where(H < 120, torch.cat([X, C, zeros], dim=1), rgb)
    rgb = torch.where(H < 60, torch.cat([C, X, zeros], dim=1), rgb)

    return rgb + m

def visualize_buffer_gradients(gradients, heatmap=False, maximum=None, minimum=None):
    
    # Compute a heatmap by mapping the hue from blue to red according to the value of gradients
    if heatmap:
        # Preprocess and compute magnitude
        b, c, h, w = gradients.shape
        gradients = torch.abs(gradients).clamp(0, 1e25)
        gradients = torch.sqrt(torch.sum(gradients * gradients, dim=1, keepdim=True))

        # Fit within a range
        if maximum is None:
            maximum = torch.max(gradients) / 10
        if minimum is None:
            minimum = 0.0
        gradients = ((gradients - minimum) / (maximum - minimum)).clamp(0, 1)

        hmap = torch.ones_like(gradients.repeat(1, 3, 1, 1))
        hue = (1 - gradients) * 0.7 # 0 is red, 0.7 is blue
        hmap[:, 0] = hue[:, 0]
        hmap = hsv_to_rgb(hmap)
        return hmap
    else:
        # Fit within a range
        if maximum is None:
            maximum = torch.max(gradients) / 10
        if minimum is None:
            minimum = 0.0
        gradients = ((gradients - minimum) / (maximum - minimum)).clamp(0, 1)

        if gradients.shape[1] == 1:
            gradients = gradients.repeat(1, 3, 1, 1)

    return gradients
    