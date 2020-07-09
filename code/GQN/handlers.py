import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from GQN.representation import TowerNet, PyramidNet
from GQN.generator import GQNGenerator, UnetGenerator, PixelCNN, SumNet
from util.datasets import get_buffer_length

# Helper functions for extracting a set of passes from the pass_dict
def create_query_tensor(pass_dict, passes, size):
    if passes != []:
        i = 0
        for key in passes:
            if key == "indirect":
                pass_dict[key] = pass_dict[key].clamp(0, 1e25)
            if key == "mirror_hit" or key == "mirror" or key == "mirror_normal":
                # Mask mirror buffers
                if "roughness" in passes:
                    mask = pass_dict["roughness"] > 0.75
                    pass_dict[key] = torch.where(mask, torch.zeros_like(pass_dict[key]), pass_dict[key]).detach()
            
            if i == 0: tensor = pass_dict[key]
            else: tensor = torch.cat([tensor, pass_dict[key]], dim=1)
            i += 1

        b, *shp = tensor.shape
        tensor = F.interpolate(tensor, size=(size, size))
    else:
        element = next(iter(pass_dict.values()))
        tensor = torch.zeros([element.shape[1], 0, size, size], dtype=element.dtype, device=element.device)
    return tensor

def create_observation_tensor(pass_dict, passes, size):
    if passes != []:
        i = 0
        for key in passes:
            if key == "indirect":
                pass_dict[key] = pass_dict[key].clamp(0, 1e25)
            if key == "mirror_hit" or key == "mirror" or key == "mirror_normal":
                # Mask mirror buffers
                if "roughness" in passes:
                    mask = pass_dict["roughness"] > 0.75
                    pass_dict[key] = torch.where(mask, torch.zeros_like(pass_dict[key]), pass_dict[key]).detach()
            
            if i == 0: tensor = pass_dict[key]
            else: tensor = torch.cat([tensor, pass_dict[key]], dim=2)
            i += 1

        b, n, *shp = tensor.shape
        tensor = tensor.view((b * n, *shp))
        tensor = F.interpolate(tensor, size=(size, size))
        tensor = tensor.view((b, n, shp[0], size, size))
    else:
        element = next(iter(pass_dict.values()))
        tensor = torch.zeros([element.shape[1], element.shape[2], 0, size, size], dtype=element.dtype, device=element.device)
    return tensor

# Create final representation if representations are concatenated
def create_representation(repr_list, representations):
    for j in range(len(repr_list)):
        index = repr_list[j]
        if j == 0:
            r = representations[index]
        else:
            r2 = representations[index]
            
            # Match representation dimensions
            if r2.shape[-1] < r.shape[-1]:
                b, n, *shp = r2.shape
                r2 = r2.view((b * n, *shp))
                r2 = F.interpolate(r2, size=(r.shape[-1], r.shape[-1]))
                r2 = tensor.view((b, n, shp[0], r.shape[-1], r.shape[-1]))
            elif r.shape[-1] < r2.shape[-1]:
                b, n, *shp = r.shape
                r = r2.view((b * n, *shp))
                r = F.interpolate(r, size=(r2.shape[-1], r2.shape[-1]))
                r = tensor.view((b, n, shp[0], r2.shape[-1], r2.shape[-1]))

            r = torch.cat([r, r2], dim=1)
    if len(repr_list) == 0:
        r = torch.zeros([representations[0].shape[0], 0, 1, 1], device=representations[0].device, dtype=representations[0].dtype)
    return r

"""
A handler that handles inputs and outputs of generators
"""

class GeneratorHandler(nn.Module):
    def __init__(self, settings):
        super(GeneratorHandler, self).__init__()

        self.generators = nn.ModuleList([])
        self.sizes = []
        self.input_passes = []
        self.output_passes = []
        self.repr_lists = []
        self.observation_passes = []
        self.override_sample = []
        self.override_train = []
        self.detach = []
        self.loss_discard = []

        if settings.multi_gpu:
            self.parallel = True
        else:
            self.parallel = False

        # Read the settings of each generator
        for i in range(len(settings.model.generators)):
            if settings.model.generators[i].type == "Unet":
                self.generators.append(UnetGenerator(settings, i))
            elif settings.model.generators[i].type == "PixelCNN":
                self.generators.append(PixelCNN(settings, i))
            elif settings.model.generators[i].type == "Sum":
                self.generators.append(SumNet(settings, i))
            elif settings.model.generators[i].type == "Copy":
                index = settings.model.generators[i].settings.index
                self.generators.append(self.generators[index])
            else:
                self.generators.append(GQNGenerator(settings, i))

            if self.parallel:
                self.generators[-1] = torch.nn.DataParallel(self.generators[-1])

            self.sizes.append(settings.model.generators[i].render_size)
            self.input_passes.append(settings.model.generators[i].query_passes)
            self.output_passes.append(settings.model.generators[i].output_passes)
            self.repr_lists.append(settings.model.generators[i].representations)
            self.override_sample.append(settings.model.generators[i].override_sample)
            self.loss_discard.append(settings.model.generators[i].discard_loss)
            self.override_train.append(settings.model.generators[i].override_train)
            self.detach.append(settings.model.generators[i].detach)

        self.observation_passes = []
        for i in range(len(settings.model.representations)):
            self.observation_passes += settings.model.representations[i].observation_passes

    def separate_output(self, output, index):
        buffers = {}

        offset = 0
        for buffer in self.output_passes[index]:
            buf_len = get_buffer_length(buffer)
            buffers[buffer] = output[:, offset:(offset + buf_len)]
            offset += buf_len

        return buffers

    def forward(self, observation_passes, observation_poses, query_passes, query_poses, representations, iteration):
        loss = 0
        outputs = []
        metrics = {}
        query_pass_state = {}
        for key in query_passes:
            query_pass_state[key] = query_passes[key]
        
        # Loop through each generator, get its inputs and forward
        for i in range(len(self.generators)):
            input_tensor = create_query_tensor(query_pass_state, self.input_passes[i], self.sizes[i])
            output_tensor = create_query_tensor(query_passes, self.output_passes[i], self.sizes[i])
            observation_tensor = create_observation_tensor(observation_passes, self.observation_passes, self.sizes[i])
            
            r = create_representation(self.repr_lists[i], representations)
            
            output = self.generators[i](input_tensor, output_tensor, observation_tensor, query_poses, observation_poses, r, iteration)

            # Average losses
            for key in output["losses"]:
                value = output["losses"][key].mean()
                if not self.loss_discard[i]:
                    loss += value
                metrics[key] = value.item()

            # Process output and put into buffers
            output_buffers = self.separate_output(output["generated"], i)
            buffers = {}
            for key in output_buffers:
                buffers[key] = output_buffers[key]
            outputs.append(buffers)

            if self.override_train[i]:
                for key in output_buffers:
                    if self.detach[i]:
                        output_buffers[key] = output_buffers[key].detach()

                    query_pass_state[key] = output_buffers[key]

        return outputs, loss, metrics

    def sample(self, observation_passes, observation_poses, query_passes, query_poses, representations):
        outputs = []
        query_pass_state = {}
        for key in query_passes:
            query_pass_state[key] = query_passes[key]

        size = self.sizes[0]
        if len(self.input_passes[-1]) != 0:
            size = query_passes[self.input_passes[-1][0]].shape[2]

        for i in range(len(self.generators)):
            input_tensor = create_query_tensor(query_pass_state, self.input_passes[i], size)
            observation_tensor = create_observation_tensor(observation_passes, self.observation_passes, self.sizes[i])
            
            r = create_representation(self.repr_lists[i], representations)

            if self.parallel:
                y = self.generators[i].module.sample(input_tensor, observation_tensor, query_poses, observation_poses, r)
            else:
                y = self.generators[i].sample(input_tensor, observation_tensor, query_poses, observation_poses, r)
            
            # Process output and put into buffers
            output_buffers = self.separate_output(y, i)
            buffers = {}
            for key in output_buffers:
                buffers[key] = output_buffers[key]
            outputs.append(buffers)

            if self.override_sample[i]:
                for key in output_buffers:
                    query_pass_state[key] = output_buffers[key]

        return outputs

'''
Class for handling multiple representations
'''

class RepresentationHandler(nn.Module):
    def __init__(self, settings, iteration):
        super(RepresentationHandler, self).__init__()

        self.representations = nn.ModuleList([])
        self.sizes = []
        self.passes = []
        self.detach = []
        self.aggr_funcs = []

        if settings.multi_gpu:
            self.parallel = True
        else:
            self.parallel = False

        for i in range(len(settings.model.representations)):
            self.sizes.append(settings.model.representations[i].render_size)
            self.detach.append(settings.model.representations[i].detach)
            self.passes.append(settings.model.representations[i].observation_passes)
            self.aggr_funcs.append(settings.model.representations[i].aggregation_func)

            if settings.model.representations[i].type == "Pool":
                self.representations.append(TowerNet(settings, i, pool=True))
            elif settings.model.representations[i].type == "Pyramid":
                self.representations.append(PyramidNet(settings, i))
            else:
                self.representations.append(TowerNet(settings, i))

            self.representations[-1].iteration = iteration

            if self.parallel:
                self.representations[-1] = torch.nn.DataParallel(self.representations[-1])

    def forward(self, observation_passes, query_passes, observation_poses, query_poses, factor):
        outputs = []
        loss = 0

        for i in range(len(self.representations)):
            # Get passes for observations
            observation_images = create_observation_tensor(observation_passes, self.passes[i], self.sizes[i])

            rk = self.representations[i](observation_images, observation_poses, factor)
            if self.parallel:
                r, metrics, r_loss = self.representations[i].module.aggregate(rk, factor)
            else:
                r, metrics, r_loss = self.representations[i].aggregate(rk, factor)

            if self.detach[i]:
                r = r.detach()
                rk = rk.detach()

            for key in r_loss:
                value = r_loss[key].mean()
                loss += value
                metrics[key] = loss.item()

            outputs.append(r)

        return outputs, metrics, loss

    # Corrects gradients of representations if partitioning is used (avg-based)
    def correct_latent_gradients(self, factor):

        for i in range(len(self.representations)):
            if not self.detach[i]:
                if self.parallel:
                    self.representations[i].module.correct_latent_gradients(factor)
                else:
                    self.representations[i].correct_latent_gradients(factor)

    # Freeze gradients so they are not removed during backprop
    def retain_gradients(self):
        for i in range(len(self.representations)):
            if not self.detach[i]:
                if self.parallel:
                    self.representations[i].module.representation.retain_grad()
                    self.representations[i].module.deltas.retain_grad()
                else:
                    self.representations[i].representation.retain_grad()
                    self.representations[i].deltas.retain_grad()






