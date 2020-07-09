import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from util.datasets import get_buffer_length

# Base class for representation networks
class RepresentationNet(nn.Module):
    def __init__(self, settings, index):
        super(RepresentationNet, self).__init__()
        self.aggr_func = settings.model.representations[index].aggregation_func

        self.latent_separation = settings.latent_separation
        self.adaptive_separation = settings.adaptive_separation
        self.empty_partition = settings.empty_partition
        self.partition_loss = settings.partition_loss
        self.representation = torch.zeros([], requires_grad=True)
        self.gradient_reg = settings.model.representations[index].gradient_reg

        # Adaptive separation widths
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda else "cpu")
        if self.empty_partition:
            deltas = torch.ones(4, device=device).float()
        else:
            deltas = torch.ones(3, device=device).float()
        self.deltas = torch.nn.Parameter(deltas.detach())
        #self.deltas = torch.nn.Parameter(torch.rand(3, device=device).float())
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.iteration = 0

        self.start_sigmoid = settings.model.representations[index].start_sigmoid
        self.end_sigmoid = settings.model.representations[index].end_sigmoid
        self.sharp_sigmoid = settings.model.representations[index].sharp_sigmoid
        self.sharpening = settings.model.representations[index].sharpening
        self.final_sharpening = settings.model.representations[index].final_sharpening     

    # Has to be overriden by classes inheriting
    def prop(self, x, v):
        return None

    def create_sigmoid_mask(self, x, a, b, m):
        # Normal sigmoid "basket"
        sigmoid1 = 1 - self.sigmoid(m * (x - a))
        sigmoid2 = self.sigmoid(m * (x - b))
        normal_mask = sigmoid1 + sigmoid2

        # -1 shifted sigmoid "basket"
        sigmoid3 = 1 - self.sigmoid(m * (x - (a - 1)))
        sigmoid4 = self.sigmoid(m * (x - (b - 1)))
        shifted_mask1 = sigmoid3 + sigmoid4

        # +1 shifted sigmoid "basket"
        sigmoid5 = 1 - self.sigmoid(m * (x - (a + 1)))
        sigmoid6 = self.sigmoid(m * (x - (b + 1)))
        shifted_mask2 = sigmoid5 + sigmoid6

        return normal_mask * shifted_mask1 * shifted_mask2

    # Function to aggregate representations
    def forward(self, observation_images, observation_poses, factor):
        # Compute representation for each observation
        batch_size, num_observations, *img_dims = observation_images.shape
        _, _, *pose_dims = observation_poses.shape
        x = observation_images.view((-1, *img_dims))      

        v = observation_poses.view((-1, *pose_dims))
        rk = self.prop(x, v)
        _, *r_dims = rk.shape
        rk = rk.view((batch_size, num_observations, *r_dims))

        return rk

    def get_multiplier(self):
        multiplier = max(self.start_sigmoid + (self.end_sigmoid - self.start_sigmoid) * self.iteration / self.sharpening, self.start_sigmoid)
        if self.iteration > self.sharpening:
            multiplier = max(self.end_sigmoid + (self.sharp_sigmoid - self.end_sigmoid) * (self.iteration - self.sharpening) / self.final_sharpening, self.end_sigmoid)
            multiplier = min(self.sharp_sigmoid, multiplier)
        return multiplier

    def aggregate(self, rk, factor):
        batch_size, num_observations, *r_dims = rk.shape
        metrics = {}
        loss = {}

        if self.latent_separation and self.adaptive_separation:
            r_dim = rk.shape[2]
            percentages = self.softmax(self.deltas)
            splits = torch.cumsum(percentages, 0)
            multiplier = self.get_multiplier()
            interp = torch.ones(r_dim, dtype=rk.dtype, device=rk.device) / r_dim
            interp = torch.cumsum(interp, 0)

        if self.latent_separation and self.adaptive_separation and factor != -1:
            # If average-based partitioning -- compute means across batch
            # and replace the current values
            # Otherwise average across independent representations only
            metrics["lighting_percentage"] = percentages[0].item()
            metrics["geometry_percentage"] = percentages[1].item()
            metrics["material_percentage"] = percentages[2].item() 
            if self.empty_partition:
                metrics["empty_percentage"] = percentages[3].item()
                metrics["absolute_rsize"] = (1 - percentages[3].item()) * r_dim

            batch_mean = torch.mean(rk, dim=[0, 1], keepdim=True).view((-1, *rk.shape[2:])).repeat(batch_size, 1, 1, 1)
            observation_mean = torch.mean(rk, dim=1)

            # Create masks where 0 means use observation_mean and 1 means batch_mean
            self.iteration += 1
            if factor == 0:
                mask = self.create_sigmoid_mask(interp, 0, splits[0], multiplier)
            elif factor == 1:
                mask = self.create_sigmoid_mask(interp, splits[0], splits[1], multiplier)
            else:
                mask = self.create_sigmoid_mask(interp, splits[1], splits[2], multiplier)
            mask = mask.view((1, r_dim, 1, 1))
            self.representation = (1 - mask) * observation_mean + mask * batch_mean

            if self.empty_partition:
                loss["partition_loss"] = self.partition_loss * r_dim / 256 * (1 - percentages[3])

        elif self.latent_separation and factor != -1:
            r_dim = rk.shape[2]
            split_dim = r_dim // 3
            ones = torch.ones([1, r_dim, 1, 1], device=rk.device, dtype=rk.dtype)
            zeros = torch.zeros_like(ones)
            indices = torch.cumsum(ones, dim=1)
            batch_mean = torch.mean(rk, dim=[0, 1], keepdim=True).view((-1, *rk.shape[2:])).repeat(batch_size, 1, 1, 1)
            observation_mean = torch.mean(rk, dim=1)
            if factor == 0:
                mask = torch.where(indices < split_dim, zeros, ones)
            elif factor == 1:
                mask = torch.where((indices >= split_dim) & (indices < (2 * split_dim)), zeros, ones)
            else:
                mask = torch.where(indices >= (split_dim * 2), zeros, ones)
            self.representation = (1 - mask) * observation_mean + mask * batch_mean
        else:
            # For each scene, sum representations
            if self.aggr_func == 'max':
                self.representation = torch.max(rk, dim=1)[0]
            else:
                self.representation = torch.mean(rk, dim=1)

        if self.latent_separation and self.empty_partition and self.adaptive_separation:
            zero_mask = self.create_sigmoid_mask(interp, splits[2], 1, multiplier)
            zero_mask = zero_mask.view((1, zero_mask.shape[0], 1, 1)).clamp(0, 1)
            self.representation = self.representation * zero_mask

        return self.representation, metrics, loss

    # Average-based disentangling gradient correction
    # Replace gradients with the difference of the representation from the mean
    def correct_latent_gradients(self, factor):
        if not self.latent_separation or factor == -1:
            return

        if self.gradient_reg == 0.0:
            return

        r_dim = self.representation.shape[1]
        interp = torch.ones(r_dim, dtype=self.representation.grad.dtype, device=self.representation.grad.device) / r_dim
        interp = torch.cumsum(interp, 0)
        if self.adaptive_separation:
            splits = torch.cumsum(self.softmax(self.deltas), 0)
            multiplier = self.get_multiplier()
        else:
            splits = torch.tensor([0.33, 0.66, 1.0], dtype=self.representation.dtype, device=self.representation.device)
            multiplier = 100000.0

        grad_mean = torch.mean(self.representation.grad, dim=0)

        if factor == 0:
            mask = self.create_sigmoid_mask(interp, 0, splits[0], multiplier)
        elif factor == 1:
            mask = self.create_sigmoid_mask(interp, splits[0], splits[1], multiplier)
        else:
            mask = self.create_sigmoid_mask(interp, splits[1], splits[2], multiplier)
        mask = mask.view((1, r_dim, 1, 1))

        self.representation.grad = self.gradient_reg * mask * (grad_mean - self.representation.grad) + (1 - mask * self.gradient_reg) * self.representation.grad

# Pyramid representation
class PyramidNet(RepresentationNet):
    def __init__(self, settings, index):
        super(PyramidNet, self).__init__(settings, index)
        # Input: (image_dim + pose_dim)x64x64
        self.v_dim = settings.model.pose_dim
        self.r_dim = settings.model.representations[index].representation_dim
        self.w = settings.model.representations[index].render_size

        self.x_dim = 0
        for buffer in settings.model.representations[index].observation_passes:
            self.x_dim += get_buffer_length(buffer)

        self.filters = 32
        # [filters, h/2, w/2]
        self.in_conv = nn.Conv2d(self.x_dim + self.v_dim, self.filters, 2, stride=2)
        
        self.conv_list = nn.ModuleList([])
        i = 1
        d = self.w
        
        # Half dimensions every iteration and double num filters 
        while d > 16:
            self.conv_list.append(nn.Conv2d(self.filters * i, self.filters * (i * 2), 2, stride=2))
            i = i * 2
            d = d // 2
        
        # Output: [r_dim, 1, 1]
        self.out_conv = nn.Conv2d(self.filters * i, self.r_dim, 8, stride=8)
        
    def prop(self, x, v):
        # [v_dim, 1, 1] -> [v_dim, h, w]
        v = v.repeat(1, 1, self.w, self.w)
        
        X = torch.cat((x, v), dim=1)
        X = F.relu(self.in_conv(X))

        for i in range(len(self.conv_list)):
            X = F.relu(self.conv_list[i](X))

        X = F.relu(self.out_conv(X))
        return X

    def get_dims(self):
        return [self.r_dim, 1, 1]

# Tower and Pool representation networks
class TowerNet(RepresentationNet):
    def __init__(self, settings, index, pool=False):
        super(TowerNet, self).__init__(settings, index)
        self.pool = pool
        self.r_dim = settings.model.representations[index].representation_dim
        self.v_dim = settings.model.pose_dim
        self.out_dim = settings.model.representations[index].render_size // 4

        self.x_dim = 0
        for buffer in settings.model.representations[index].observation_passes:
            self.x_dim += get_buffer_length(buffer)

        # [r, h // 2, w // 2]
        self.conv1 = nn.Conv2d(self.x_dim, self.r_dim, 2, stride=2)
        # [r // 2, h // 2, w // 2]
        self.conv2 = nn.Conv2d(self.r_dim, self.r_dim // 2, 3, stride=1, padding=1)
        # [r // 2, h // 4, w // 4]
        self.conv3 = nn.Conv2d(self.r_dim // 2, self.r_dim, 2, stride=2)

        self.skip_conv1 = nn.Conv2d(self.r_dim, self.r_dim, 2, stride=2)
        self.skip_conv2 = nn.Conv2d(self.r_dim + self.v_dim, self.r_dim, 3, stride=1, padding=1)

        # [r // 2, h // 4, w // 4] 
        self.conv4 = nn.Conv2d(self.r_dim + self.v_dim, self.r_dim // 2, 3, stride=1, padding=1)
        # [r, h // 4, w // 4]
        self.conv5 = nn.Conv2d(self.r_dim // 2, self.r_dim, 3, stride=1, padding=1)
        # [r, h // 4, w // 4]
        self.conv6 = nn.Conv2d(self.r_dim, self.r_dim, 1, stride=1)

        # [r, 1, 1]
        self.pooling = nn.AvgPool2d(self.out_dim)
        
    def prop(self, x, v):
        # [v_dim, 1, 1] -> [v_dim, h//4, w//4]
        v = v.repeat(1, 1, self.out_dim, self.out_dim)

        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.skip_conv1(skip_in))
        x = F.relu(self.conv2(skip_in))
        x = F.relu(self.conv3(x)) + skip_out

        skip_in = torch.cat([x, v], dim=1)
        skip_out = F.relu(self.skip_conv2(skip_in))

        x = F.relu(self.conv4(skip_in))
        x = F.relu(self.conv5(x)) + skip_out
        x = F.relu(self.conv6(x))
        if self.pool:
            x = self.pooling(x)
        return x

    def get_dims(self):
        if self.pool:
            return [self.r_dim, 1, 1]
        else:
            return [self.r_dim, self.out_dim, self.out_dim]
