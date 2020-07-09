# Classes for different settings, config file values are stored into these
class Settings():
    def __init__(self):
        self.logging = True
        self.seed = -1
        self.checkpoint_interval = 500
        self.test_interval = 25
        self.iterations = 1000000
        self.multi_gpu = True
        
        self.batch_size = 36
        self.test_batch_size = 16
        self.dataset = "sphere_grid"
        self.cached_dataset = False
        self.test_data_dir = ""
        self.train_data_dir = ""
        self.job_name = ''
        self.job_group = 'Undefined'
        self.antialias = False
        self.samples_per_pixel = 64
        self.views_per_scene = 5
        self.latent_separation = False
        self.adaptive_separation = True
        self.empty_partition = False
        self.partition_loss = 0.01
        self.random_num_views = True
        self.model = ModelSettings()

    def __repr__(self):
        return str(self.__dict__)

class ModelSettings():
    def __init__(self):
        self.representations = [RepresentationSettings()]
        self.generators = [GeneratorSettings()]
        self.pose_dim = 16
        self.output_pass = "beauty"

    def __repr__(self):
        return str(self.__dict__)

class RepresentationSettings():
    def __init__(self):
        self.type = "Tower"
        self.detach = False
        self.aggregation_func = "mean"
        self.observation_passes = ["beauty"]
        self.representation_dim = 256
        self.render_size = 64
        self.start_sigmoid = 5.0
        self.end_sigmoid = 100.0
        self.sharp_sigmoid = 100.0
        self.sharpening = 5e5
        self.final_sharpening = 1e5
        self.gradient_reg = 0.0

    def __repr__(self):
        return str(self.__dict__)

class GeneratorSettings():
    def __init__(self):
        self.type = "GQN"
        self.query_passes = []
        self.output_passes = ["beauty"]
        self.representations = [0]
        self.render_size = 64
        self.override_sample = False
        self.discard_loss = False
        self.override_train = False
        self.detach = False
        self.settings = GQNSettings()

    def __repr__(self):
        return str(self.__dict__)

class GQNSettings():
    def __init__(self):
        self.cell_type = "LSTM"
        self.latent_dim = 3
        self.state_dim = 128
        self.core_count = 12
        self.downscaling = 4
        self.weight_sharing = False
        self.upscale_sharing = True

    def __repr__(self):
        return str(self.__dict__)

class UnetSettings():
    def __init__(self):
        self.loss = ["l1", "ssim"]
        self.end_relu = True

    def __repr__(self):
        return str(self.__dict__)

class PixelCNNSettings():
    def __init__(self):
        self.loss = ["l1", "ssim"]
        self.layers = 8
        self.hidden_units = 512
        self.end_relu = True
        self.propagate_buffers = False
        self.propagate_representation = True
        self.propagate_viewpoint = True

    def __repr__(self):
        return str(self.__dict__)

class SumSettings():
    def __init__(self):
        self.dummy = {}

    def __repr__(self):
        return str(self.__dict__)


# This is a bit weird, but this does not create a new network
# It only refers to another network.
class CopySettings():
    def __init__(self):
        self.index = 0

    def __repr__(self):
        return str(self.__dict__)


