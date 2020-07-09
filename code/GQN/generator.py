import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import pytorch_ssim
import torchvision
from GQN.core import LSTMCore, GRUCore
from util.datasets import get_buffer_length
import copy

"""
Implementation of the generator used in the GQN paper
can also replace LSTM cells with GRU cells
"""
class GQNGenerator(nn.Module):
    def __init__(self, settings, index):
        super(GQNGenerator, self).__init__()
        self.core_count = settings.model.generators[index].settings.core_count
        self.share = settings.model.generators[index].settings.weight_sharing
        self.upscale_share = settings.model.generators[index].settings.upscale_sharing
        self.index = index

        # Set dimensions
        self.h_dim = settings.model.generators[index].settings.state_dim
        self.v_dim = settings.model.pose_dim
        self.z_dim = settings.model.generators[index].settings.latent_dim
        self.height = settings.model.generators[index].render_size
        self.width = settings.model.generators[index].render_size
        self.representations = settings.model.generators[index].representations

        self.x_dim = 0
        for buffer in settings.model.generators[index].query_passes:
            self.x_dim += get_buffer_length(buffer)

        self.r_dim = 0
        for r in self.representations:
            self.r_dim += settings.model.representations[r].representation_dim
        
        self.scaling = settings.model.generators[index].settings.downscaling
        self.downscaled_height = self.height // self.scaling
        self.downscaled_width = self.width // self.scaling
        self.out_dim = 0
        for buffer in settings.model.generators[index].output_passes:
            self.out_dim += get_buffer_length(buffer)

        inference_dim = self.v_dim + self.r_dim + self.out_dim + self.h_dim
        generator_dim = self.v_dim + self.r_dim + self.z_dim + self.x_dim

        if settings.model.generators[index].settings.cell_type == "GRU":
            self.core = GRUCore
            self.gru = True
        else:
            self.core = LSTMCore
            self.gru = False

        # Set up computational cores
        if self.share:
            self.inference = self.core(inference_dim, self.h_dim)
            self.generator = self.core(generator_dim, self.h_dim)
        else:
            self.inference = nn.ModuleList([self.core(inference_dim, self.h_dim) for _ in range(self.core_count)])
            self.generator = nn.ModuleList([self.core(generator_dim, self.h_dim) for _ in range(self.core_count)])

        # These convolutions are used to model the normal distributions
        self.prior = nn.Conv2d(self.h_dim, self.z_dim * 2, 5, padding=2)
        self.posterior = nn.Conv2d(self.h_dim, self.z_dim * 2, 5, padding=2)
        
        # Final convolution on the canvas vairable
        self.observation = nn.Conv2d(self.h_dim, self.out_dim, 1)

        # Scaling convolutions performed on input and canvas of cores
        # No bias because we don't want to add information that is not already in the input
        if self.x_dim > 0:
            self.downsample = nn.Conv2d(self.x_dim, self.x_dim, self.scaling, stride=self.scaling, padding=0, bias=False)
        if self.upscale_share:
            self.upsample = nn.ConvTranspose2d(self.h_dim, self.h_dim, self.scaling, stride=self.scaling, padding=0, bias=False)
        else:
            self.upsample = nn.ModuleList([nn.ConvTranspose2d(self.h_dim, self.h_dim, self.scaling, stride=self.scaling, padding=0, bias=False) for _ in range(self.core_count)])

    def forward(self, x, y, o, v, p, r, iteration, additional=None):
        """
        h_0, c_0, u_0 = (0, 0, 0)
        for l in L:
            compute prior factor from normal distribution
            update inference state
            compute posterior factor from normal distribution
            sample posterior to get z_l
            update generator state
            add to ELBO KL computation
        """
        
        kl = 0
        batch_size = y.shape[0]
        inf_state_h = torch.zeros([batch_size, self.h_dim, self.downscaled_height, self.downscaled_width], dtype=x.dtype, device=x.device)
        gen_state_h = torch.zeros([batch_size, self.h_dim, self.downscaled_height, self.downscaled_width], dtype=x.dtype, device=x.device)
        
        if not self.gru:
            inf_state_c = torch.zeros([batch_size, self.h_dim, self.downscaled_height, self.downscaled_width], dtype=x.dtype, device=x.device)
            gen_state_c = torch.zeros([batch_size, self.h_dim, self.downscaled_height, self.downscaled_width], dtype=x.dtype, device=x.device)

        canvas = torch.zeros([batch_size, self.h_dim, self.height, self.width], dtype=x.dtype, device=x.device)

        r = r.repeat(1, 1, self.downscaled_height // r.shape[2], self.downscaled_width // r.shape[3])
        v = v.repeat(1, 1, self.downscaled_height // v.shape[2], self.downscaled_width // v.shape[3])
        
        if self.x_dim > 0: x_ds = self.downsample(x)
        else: x_ds = torch.zeros([batch_size, 0, self.downscaled_height, self.downscaled_width], dtype=x.dtype, device=x.device)
        y_ds = F.interpolate(y, size=(self.downscaled_height, self.downscaled_width))
        
        for l in range(self.core_count):
            # Compute prior distribution by extracting mean and logstd
            p_mean, p_logstd = torch.chunk(self.prior(gen_state_h), 2, dim=1)
            prior_distr = D.Normal(p_mean, F.softplus(p_logstd))

            # Inference state update
            inf_input = torch.cat([gen_state_h, y_ds, v, r], dim=1)

            # Select cell
            if self.share:
                cell = self.inference
            else:
                cell = self.inference[l]

            if self.gru:
                inf_state_h = cell(inf_input, inf_state_h)
            else:
                inf_state_h, inf_state_c = cell(inf_input, inf_state_h, inf_state_c)
            # Compute posterior distribution by extracting mean and logstd
            q_mean, q_logstd = torch.chunk(self.posterior(inf_state_h), 2, dim=1)
            posterior_distr = D.Normal(q_mean, F.softplus(q_logstd))

            # Generator state update
            z = posterior_distr.rsample() # rsample makes sample differentiable
            gen_input = torch.cat([v, r, z, x_ds], dim=1)

            if self.share:
                cell = self.generator
            else:
                cell = self.generator[l]

            if self.gru:
                gen_state_h = cell(gen_input, gen_state_h)
            else:
                gen_state_h, gen_state_c = cell(gen_input, gen_state_h, gen_state_c)

            # Update canvas and KL divergence
            if self.upscale_share:
                canvas += self.upsample(gen_state_h)
            else:
                canvas += self.upsample[l](gen_state_h)
            kl += D.kl.kl_divergence(posterior_distr, prior_distr)

        y_mean = self.observation(canvas)

        # Compute ELBO
        sigma = max(0.7 + (2.0 - 0.7) * (1 - iteration / 2e5), 0.7)
        ll = D.Normal(y_mean, sigma).log_prob(torch.log1p(y))
        ll = torch.sum(ll, dim=[1,2,3])
        kl = torch.sum(kl, dim=[1,2,3])

        losses = {}
        losses["generator" + str(self.index) + "_kl"] = kl / (self.width * 25)
        losses["generator" + str(self.index) + "_ll"] = -ll / (self.width * 25)
        
        outputs = {}
        outputs["generated"] = torch.exp(y_mean) - 1
        outputs["losses"] = losses
        
        return outputs

    def sample(self, x, o, v, p, r):
        batch_size = r.shape[0]
        w = x.shape[2]
        dh = w // self.scaling
        gen_state_h = torch.zeros([batch_size, self.h_dim, dh, dh], dtype=v.dtype, device=v.device)
        
        if not self.gru:
            gen_state_c = torch.zeros([batch_size, self.h_dim, dh, dh], dtype=v.dtype, device=v.device)

        canvas = torch.zeros([batch_size, self.h_dim, w, w], dtype=v.dtype, device=v.device)
        if self.x_dim > 0: x_ds = self.downsample(x)
        else: x_ds = torch.zeros([batch_size, 0, dh, dh], dtype=x.dtype, device=x.device)
        r = r.repeat(1, 1, dh // r.shape[2], dh // r.shape[3])
        v = v.repeat(1, 1, dh // v.shape[2], dh // v.shape[3])

        for l in range(self.core_count):
            # Compute prior distribution by extracting mean and logstd
            p_mean, p_logstd = torch.chunk(self.prior(gen_state_h), 2, dim=1)
            prior_distr = D.Normal(p_mean, F.softplus(p_logstd))
            
            # Compute generator output
            z = prior_distr.sample()

            gen_input = torch.cat([v, r, z, x_ds], dim=1)
            
            if self.share:
                cell = self.generator
            else:
                cell = self.generator[l]

            if self.gru:
                gen_state_h = cell(gen_input, gen_state_h)
            else:
                gen_state_h, gen_state_c = cell(gen_input, gen_state_h, gen_state_c)

            if self.upscale_share:
                canvas += self.upsample(gen_state_h)
            else:
                canvas += self.upsample[l](gen_state_h)

        x_mean = self.observation(canvas)
        return torch.exp(x_mean) - 1

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        features = torchvision.models.vgg19(pretrained=True).features
        self.feature1 = torch.nn.Sequential()
        for i in range(0, 4):
            self.feature1.add_module(str(i), features[i])
        
        self.feature2 = torch.nn.Sequential()
        for i in range(4, 9):
            self.feature2.add_module(str(i), features[i])
        
        self.feature3 = torch.nn.Sequential()
        for i in range(9, 16):
            self.feature3.add_module(str(i), features[i])
        
        self.feature4 = torch.nn.Sequential()
        for i in range(16, 23):
            self.feature4.add_module(str(i), features[i])

        self.feature5 = torch.nn.Sequential()
        for i in range(23, 32):
            self.feature5.add_module(str(i), features[i])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        output = []
        X = F.interpolate(X, size=(224, 224), mode='bilinear')
        X = self.feature1(X)
        output.append(X)
        X = self.feature2(X)
        output.append(X)
        X = self.feature3(X)
        output.append(X)
        X = self.feature4(X)
        output.append(X)
        X = self.feature5(X)
        output.append(X)
        return output

class UnetBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UnetBlock, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, 3, padding=1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.model(x)

class UnetGenerator(nn.Module):
    def __init__(self, settings, index):
        super(UnetGenerator, self).__init__()
        self.v_dim = settings.model.pose_dim
        self.w = settings.model.generators[index].render_size
        self.loss = settings.model.generators[index].settings.loss
        self.index = index
        self.representations = settings.model.generators[index].representations

        self.x_dim = 0
        for buffer in settings.model.generators[index].query_passes:
            self.x_dim += get_buffer_length(buffer)

        self.out_dim = 0
        for buffer in settings.model.generators[index].output_passes:
            self.out_dim += get_buffer_length(buffer)

        self.r_dim = 0
        for r in self.representations:
            self.r_dim += settings.model.representations[r].representation_dim

        # Set up architecture
        self.down_block1 = UnetBlock(self.x_dim + self.r_dim + self.v_dim, 128)
        self.pool1 = nn.MaxPool2d(2, stride=2) # 32x32
        self.down_block2 = UnetBlock(128 + self.r_dim + self.v_dim, 256)
        self.pool2 = nn.MaxPool2d(2, stride=2) # 16x16
        self.down_block3 = UnetBlock(256 + self.r_dim + self.v_dim, 512)

        self.pool3 = nn.MaxPool2d(2, stride=2) # 8x8
        self.down_block4 = UnetBlock(512 + self.r_dim + self.v_dim, 512)

        self.pool4 = nn.MaxPool2d(2, stride=2) # 4x4
        self.down_block5 = UnetBlock(512 + self.r_dim + self.v_dim, 512)

        self.pool5 = nn.MaxPool2d(2, stride=2) # 2x2
        self.down_block6 = UnetBlock(512 + self.r_dim + self.v_dim, 512)

        self.pool6 = nn.MaxPool2d(2, stride=2) # 1x1
        self.down_block7 = UnetBlock(512 + self.r_dim + self.v_dim, 512) 
        
        self.upscale4 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1) # 1x1
        self.up_block4 = UnetBlock(1024, 512)

        self.upscale5 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1) # 2x2
        self.up_block5 = UnetBlock(1024, 512)

        self.upscale6 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1) # 4x4
        self.up_block6 = UnetBlock(1024, 512)

        self.upscale1 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1) # 8x8
        self.up_block1 = UnetBlock(1024, 512)
        self.upscale2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # 16x16
        self.up_block2 = UnetBlock(512, 256)
        self.upscale3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # 32x32
        self.up_block3 = UnetBlock(256, 128)

        self.final = nn.Conv2d(128, self.out_dim, 1)

        self.l1_loss = nn.L1Loss()
        self.end_relu = settings.model.generators[index].settings.end_relu

    def eval(self, x, o, v, p, r):

        batch_size, num_observations, pose_dim, *pose_dims = p.shape
        p = p.view((batch_size, num_observations * pose_dim, *pose_dims))

        w = x.shape[2]

        r1_shape = (batch_size, self.r_dim, w, w)
        r2_shape = (batch_size, self.r_dim, w // 2, w // 2)
        r3_shape = (batch_size, self.r_dim, w // 4, w // 4)
        r4_shape = (batch_size, self.r_dim, w // 8, w // 8)
        r5_shape = (batch_size, self.r_dim, w // 16, w // 16)
        r6_shape = (batch_size, self.r_dim, w // 32, w // 32)
        r7_shape = (batch_size, self.r_dim, w // 64, w // 64)

        v1_shape = (batch_size, self.v_dim, w, w)
        v2_shape = (batch_size, self.v_dim, w // 2, w // 2)
        v3_shape = (batch_size, self.v_dim, w // 4, w // 4)
        v4_shape = (batch_size, self.v_dim, w // 8, w // 8)
        v5_shape = (batch_size, self.v_dim, w // 16, w // 16)
        v6_shape = (batch_size, self.v_dim, w // 32, w // 32)
        v7_shape = (batch_size, self.v_dim, w // 64, w // 64)

        # Go down through network
        M = torch.cat([x, r.expand(*r1_shape), v.expand(*v1_shape)], dim=1) # 64x64
        X1 = self.down_block1(M)
        X = self.pool1(X1) # 32x32

        M = torch.cat([X, r.expand(*r2_shape), v.expand(*v2_shape)], dim=1)
        X2 = self.down_block2(M)
        X = self.pool2(X2) # 16x16

        M = torch.cat([X, r.expand(*r3_shape), v.expand(*v3_shape)], dim=1)
        X3 = self.down_block3(M)
        X = self.pool3(X3) # 8x8

        M = torch.cat([X, r.expand(*r4_shape), v.expand(*v4_shape)], dim=1)
        X4 = self.down_block4(M)
        X = self.pool4(X4) # 4x4

        M = torch.cat([X, r.expand(*r5_shape), v.expand(*v5_shape)], dim=1)
        X5 = self.down_block5(M)
        X = self.pool5(X5) # 2x2

        M = torch.cat([X, r.expand(*r6_shape), v.expand(*v6_shape)], dim=1)
        X6 = self.down_block6(M)
        X = self.pool6(X6) # 1x1

        X = torch.cat([X, r.expand(*r7_shape), v.expand(*v7_shape)], dim=1)
        X = self.down_block7(X)

        # Go upwards
        X = self.upscale4(X) # 2x2
        X = torch.cat([X, X6], dim=1)
        X = self.up_block4(X)

        X = self.upscale5(X) # 4x4
        X = torch.cat([X, X5], dim=1)
        X = self.up_block5(X)

        X = self.upscale6(X) # 8x8
        X = torch.cat([X, X4], dim=1)
        X = self.up_block6(X)

        X = self.upscale1(X) # 16x16
        X = torch.cat([X, X3], dim=1)
        X = self.up_block1(X)

        X = self.upscale2(X) # 32x32
        X = torch.cat([X, X2], dim=1)
        X = self.up_block2(X)

        X = self.upscale3(X) # 64x64
        X = torch.cat([X, X1], dim=1)
        X = self.up_block3(X)
        X = self.final(X)
        if self.end_relu:
            X = F.relu(X)

        return X

    def sample(self, x, o, v, p, r):
        X = self.eval(x, o, v, p, r)
        if "logl1" in self.loss or "logssim" in self.loss:
            X = torch.exp(X) - 1
        return X

    def forward(self, x, y, o, v, p, r, iteration, additional=None):
        Y = self.eval(x, o, v, p, r)
        losses = {}

        # Different loss functions, we used logl1 + logssim
        if "logl1" in self.loss:
            logl1 = 2 * torch.abs(Y - torch.log1p(y))
            losses["generator" + str(self.index) + "_logl1"] = logl1
        if "l1" in self.loss:
            l1 = 0.5 * torch.abs(Y - y)
            losses["generator" + str(self.index) + "_l1"] = l1
        if "logssim" in self.loss:
            logssim = 1.0 - pytorch_ssim.ssim(torch.log1p(y), Y)
            losses["generator" + str(self.index) + "_logssim"] = logssim
        if "ssim" in self.loss:
            ssim = 1.0 - pytorch_ssim.ssim(y, Y)
            losses["generator" + str(self.index) + "_ssim"] = ssim
        if "nll" in self.loss:
            sigma = max(0.1 + (2.0 - 0.1) * (1 - iteration / 2e4), 0.1)
            ll = -D.Normal(Y, sigma).log_prob(y)
            losses["generator" + str(self.index) + "_nll"] = ll

        if "logl1" in self.loss or "logssim" in self.loss:
            Y = torch.exp(Y) - 1

        output = {}
        output["generated"] = Y
        output["losses"] = losses

        return output

# Pixel generator model
class PixelCNN(nn.Module):
    def __init__(self, settings, index):
        super(PixelCNN, self).__init__()
        self.v_dim = settings.model.pose_dim
        self.w = settings.model.generators[index].render_size
        self.loss = settings.model.generators[index].settings.loss
        self.index = index
        self.representations = settings.model.generators[index].representations

        self.x_dim = 0
        for buffer in settings.model.generators[index].query_passes:
            self.x_dim += get_buffer_length(buffer)

        self.r_dim = 0
        for r in self.representations:
            self.r_dim += settings.model.representations[r].representation_dim

        self.out_dim = 0
        for buffer in settings.model.generators[index].output_passes:
            self.out_dim += get_buffer_length(buffer)
 
        # Create model
        self.num_layers = settings.model.generators[index].settings.layers
        self.num_hidden = settings.model.generators[index].settings.hidden_units

        self.propagate_buffers = settings.model.generators[index].settings.propagate_buffers
        self.propagate_representation = settings.model.generators[index].settings.propagate_representation
        self.propagate_viewpoint = settings.model.generators[index].settings.propagate_viewpoint

        self.input = nn.Conv2d(self.x_dim + self.r_dim + self.v_dim, self.num_hidden, 1, stride=1)

        self.prop_dim = 0
        if self.propagate_buffers:
            self.prop_dim += self.x_dim
        if self.propagate_representation:
            self.prop_dim += self.r_dim
        if self.propagate_viewpoint:
            self.prop_dim += self.v_dim

        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv2d(self.num_hidden + self.prop_dim, self.num_hidden, 1, stride=1))

        self.final = nn.Conv2d(self.num_hidden, self.out_dim, 1, stride=1)
        self.activation = nn.LeakyReLU()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.end_relu = settings.model.generators[index].settings.end_relu
        if "vgg" in self.loss:
            self.vgg = VGG19()

    def eval(self, x, o, v, p, r):
        r_shape = (r.shape[0], r.shape[1], x.shape[2], x.shape[3])
        v_shape = (v.shape[0], v.shape[1], x.shape[2], x.shape[3])

        r = r.expand(*r_shape)
        v = v.expand(*v_shape)
 
 
        X = torch.cat([x, v, r], dim=1)
        X = self.activation(self.input(X))

        for i in range(len(self.layers)):
            if self.propagate_buffers:
                X = torch.cat([X, x], dim=1)
            if self.propagate_representation:
                X = torch.cat([X, r], dim=1)
            if self.propagate_viewpoint:
                X = torch.cat([X, v], dim=1)
            X = self.activation(self.layers[i](X))

        X = self.final(X)
        if self.end_relu:
            X = F.relu(X)
        if "bce" in self.loss:
            X = X.clamp(0, 1)

        return X

    def sample(self, x, o, v, p, r):
        X = self.eval(x, o, v, p, r)
        if "logl1" in self.loss or "logssim" in self.loss or "logbce" in self.loss or "loghuber" in self.loss or "lognll" in self.loss or "logl2" in self.loss:
            X = torch.exp(X) - 1
        return X

    def forward(self, x, y, o, v, p, r, iteration, additional=None):
        Y = self.eval(x, o, v, p, r)
        losses = {}

        # Different loss functions, we used logl1 + logssim
        if "logl1" in self.loss:
            logl1 = 2 * torch.abs(Y - torch.log1p(y))
            losses["generator" + str(self.index) + "_logl1"] = logl1
        if "l1" in self.loss:
            l1 = 2 * torch.abs(Y - y)
            losses["generator" + str(self.index) + "_l1"] = l1
        if "logl2" in self.loss:
            logl2 = 2 * torch.abs(Y - torch.log1p(y)) * torch.abs(Y - torch.log1p(y))
            losses["generator" + str(self.index) + "_l2"] = 250 * logl2
        if "logssim" in self.loss:
            logssim = 1.0 - pytorch_ssim.ssim(torch.log1p(y), Y)
            losses["generator" + str(self.index) + "_logssim"] = logssim
        if "ssim" in self.loss:
            ssim = 1.0 - pytorch_ssim.ssim(y, Y)
            losses["generator" + str(self.index) + "_ssim"] = ssim
        if "nll" in self.loss:
            sigma = max(0.1 + (2.0 - 0.1) * (1 - iteration / 2e5), 0.1)
            ll = -D.Normal(Y, sigma).log_prob(y)
            losses["generator" + str(self.index) + "_nll"] = 0.2 * ll
        if "bce" in self.loss:
            losses["generator" + str(self.index) + "_bce"] = 10 * self.bce_loss(Y, y)
        if "vgg" in self.loss:
            y_norm = torch.log1p(y).clamp(0, 1)
            Y_norm = Y.clamp(0, 1)
            target_features = self.vgg(y_norm)
            output_features = self.vgg(Y_norm)
            vgg_loss = 0
            for i in range(len(target_features)):
                vgg_loss += torch.mean((target_features[i] - output_features[i]) * (target_features[i] - output_features[i]), dim=[1, 2, 3], keepdim=True)
            losses["generator" + str(self.index) + "_vgg"] = 0.2 * vgg_loss

        if "logl1" in self.loss or "logssim" in self.loss or "logl2" in self.loss:
            Y = torch.exp(Y) - 1

        output = {}
        output["generated"] = Y
        output["losses"] = losses

        return output


# "Generator" that sums together all the input buffers
# Expects all of them to be same dimensions
# and this network has no loss
# This can be used to sum up the results of two generators such as direct and indirect outputs
# Helps with final visualization
class SumNet(nn.Module):
    def __init__(self, settings, index):
        super(SumNet, self).__init__()
        self.passes = settings.model.generators[index].query_passes

    def sample(self, x, o, v, p, r):
        buf_len = get_buffer_length(self.passes[0])
        Y = x[:, 0:buf_len, :, :]
        accum = buf_len

        for i in range(1, len(self.passes)):
            buf_len = get_buffer_length(self.passes[i])
            buffer = x[:, accum:(accum+buf_len)]
            accum += buf_len
            Y += buffer
        
        return Y

    def forward(self, x, y, o, v, p, r, iteration, additional=None):
        Y = self.sample(x, o, v, p, r)
        output = {}
        output["generated"] = Y
        output["losses"] = {}

        return output
