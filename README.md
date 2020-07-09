# Compositional Neural Scene Representations for Shading Inference

> **Compositional Neural Scene Representations for Shading Inference**\
> Jonathan Granskog<sup>1,3</sup>, Fabrice Rousselle<sup>1</sup>, Marios Papas<sup>2</sup>, Jan Novák<sup>1</sup>\
> <sup>1</sup>NVIDIA, <sup>2</sup>DisneyResearch|Studios, <sup>3</sup>ETH Zurich\
> [Website](http://granskog.xyz/shading-scene-representations) [Paper](https://jannovak.info/publications/CNSR/CNSR.pdf) [Video](https://www.youtube.com/watch?v=oPaE1TCA98Y)
>
> <p align="justify"><b>Abstract:</b> <i>We present a technique for adaptively partitioning neural scene representations. Our method disentangles lighting, material, and geometric information yielding a scene representation that preserves the orthogonality of these components, improves interpretability of the model, and allows compositing new scenes by mixing components of existing ones. The proposed adaptive partitioning respects the uneven entropy of individual components and permits compressing the scene representation to lower its memory footprint and potentially reduce the evaluation cost of the model. Furthermore, the partitioned representation enables an in-depth analysis of existing image generators. We compare the flow of information through individual partitions, and by contrasting it to the impact of additional inputs (G-buffer), we are able to identify the roots of undesired visual artifacts, and propose one possible solution to remedy the poor performance. We also demonstrate the benefits of complementing traditional forward renderers by neural representations and synthesis, e.g. to infer expensive shading effects, and show how these could improve production rendering in the future if developed further.</i></p>

## Setup

### Installing Python requirements

```
pip install -r requirements.txt
```

For the visualize-gqn script, also install

```
pip install moderngl moderngl-window imgui
```

### Optix Renderer

We provide an unoptimized path tracer for generating training data and visualizing results. It is not necessary to install the renderer if you only want to train using the pregenerated datasets. 

#### Installation

1. Download OptiX 5.1.1 from [here](https://developer.nvidia.com/designworks/optix/downloads/legacy)
2. Run the provided file and accept the license
```
sh ./NVIDIA-OptiX-SDK-5.1.1-linux64-25109142.sh
```
3. Move the output folder to the ext/ folder and rename it to Optix such that the SDK, doc and other folders are in ext/Optix/SDK for example

#### Build

This requires at least CMake version 3.1, Make and OptiX 5.1.1 installed as described above. 

```
sh ./build_renderer.sh
```

## Data

### Generating a dataset

The following command will create the JSON scene files for 90 training and 10 testing batches for the **PrimitiveRoom** dataset. 
These JSON files are used by the renderer to know the scenes to open. The default settings produce noisy images with low samples per pixel and 90+10 batches are most likely not enough to train a network properly. To change this setting, please open up the corresponding config file and find the setting called <i>samples_per_pixel</i>. Please keep in mind that generating a high-quality dataset will take a long time. 


```
python create-dataset-json-files.py --config ../configs/save_room.json --out_folder ../datasets/room_json/ --size 90 --testing_size 10
```

To produce the training data, use the following command

```
python generate-dataset.py --config ../configs/save_room.json --in_folder ../datasets/room_json/train/ --out_folder ../datasets/room/train
```

To produce the testing data, use the following command

```
python generate-dataset.py --config ../configs/save_room.json --in_folder ../datasets/room_json/test/ --out_folder ../datasets/room/test
```

To render an **ArchViz** dataset, use the config save_archviz.json instead. 

### Pre-rendered datasets

We provide our own versions of the **PrimitiveRoom** and **ArchViz** datasets. 

Download the **PrimitiveRoom** dataset from [here](https://drive.google.com/file/d/1wkd4lrn2yHUHm8VlNYwIX4kTLUFlJU2J/view?usp=sharing).

Download the **ArchViz** dataset from [here](https://drive.google.com/file/d/1LCP3OPfLYyoH3QAWyHVzHexC3pwNzF10/view?usp=sharing).

## Training

```
python train-gqn.py --config path_to_config.json
```

For example, if you just generated a **PrimitiveRoom** dataset with the previous commands, you can train on it by using the following command:

```
python train-gqn.py --config ../configs/room_beauty.json
```

To load a checkpoint, add the checkpoint argument, e.g.

```
python train-gqn.py --config ../configs/room_beauty.json --checkpoint ../checkpoints/room_beauty_1000000.pt
```

### Pre-trained checkpoints

We provide two pre-trained checkpoints, one for each dataset; both are networks trained for 1M iterations using the pixel generator and the pool encoder with adaptively learned partitioning without a null partition. These checkpoints work with the room_beauty.json and the archviz_beauty.json config files respectively. 

Download the **PrimitiveRoom** checkpoint [here](https://drive.google.com/file/d/1hVl3UnuFn2eKudgMThgv_Z9A8udR1-EB/view?usp=sharing).

Download the **ArchViz** checkpoint [here](https://drive.google.com/file/d/18NqPxG6q7rr8_evNsJzW1m9A46_DVzxw/view?usp=sharing).

## Testing

We provide a script for visualizing the results for different scenes. You can move the camera by holding down SPACE and pressing WASD or the left mouse button. A specific scene file can be loaded with the --scene_file argument. 

```
python visualize-gqn.py --config path_to_config.json --checkpoint path_to_checkpoint.pt
```

The options menu allows you to generate a new random scene or randomize the observations. You can also use the following keyboard shortcuts to perform certain actions:

* R - generates a new random scene
* O - renders a new set of observations
* J - increases number of query samples by 10
* H - reduces number of query samples by 10
* M - increases number of observation samples by 10
* N - reduces number of observation samples by 10

The large image on the left is the reference whereas the right image is the prediction. The observation images are shown below the reference image and the buffers given to the final generator are shown below the predicted image.

## Attribution

We provide a script for visualizing gradient x input attributions. It is possible to compute attributions for the representation, individual partitions or specific patches in the images. Note that this tool is specifically made for networks with three observation images. 

```
python visualize-attributions.py --config path_to_config.json --checkpoint path_to_checkpoint.pt
```

To load a specific scene file, use the --scene_file argument. You can also save your settings in the viewer and load them with the --settings argument.

The viewer launches by enabling patch-based attribution. There is a small red dot in the generated image that indicates this patch. It is initialized to be equal to one pixel in size, but can be adjusted by changing the 'Patch Size' slider. The 'Range Max' and 'Range Min' sliders determine how the attribution map is visualized in the bottom row for the observations. Finally, the 'Pixel X' and 'Pixel Y' sliders move the patch around in the image. 

On the left side, there are two boxes to choose which buffers to visualize. The top box refers to the query buffer, which is visualized on the top right side of the window, whereas the bottom box changes the buffer for the observations. 

The top has a few checkboxes. 'Activate Mean' computes the mean attribution for the whole image (slow) or for the whole representation if 'Representation Grad' is selected. 'Representation Grad' disables patch-based attribution and focuses on the scene representation instead. This allows looking at single latent variables if 'Activate Mean' is disabled. If 'Partition Mean' is enabled, then the attribution is computed for a specific partition in the representation. The 'Pixel X' slider changes this. 


## Acknowledgments

We thank the creators of the models used in the **ArchViz** dataset:
* [Jay-Artist](https://www.blendswap.com/profile/1574)
* [Benedikt Bitterli](https://benedikt-bitterli.me/resources/)
* [Wig42](https://www.blendswap.com/profile/130393)
* [MZiemys](https://www.blendswap.com/profile/259105)
* [Wolgraph](https://www.blendswap.com/profile/856965)
* [dpamplin](https://www.blendswap.com/profile/142765)
* [SlykDrako](https://www.blendswap.com/profile/324)

We also thank the developers of the following projects for sharing their work openly:

* [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
* [pybind11](https://github.com/pybind/pybind11)
* [filesystem](https://github.com/wjakob/filesystem)
* [stb](https://github.com/nothings/stb)
* [rapidjson](https://github.com/Tencent/rapidjson)
* [tinyobj](https://github.com/tinyobjloader/tinyobjloader)

## License

This code is shared with an MIT License. All external code is licensed according to the licenses found within those folders. 

## Config options

### Main options

- *logging* : Write Tensorboard plots to runs/
- *seed* : Random seed
- *train_data_dir* : Path to training data
- *test_data_dir* : Path to testing data
- *job_name* : Name of job, used to name checkpoints and Tensorboard logging directory
- *job_group* : Not used, but can be used on a cluster to group jobs
- *checkpoint_interval* : How often to save a checkpoint
- *test_interval* : How often metrics are saved to Tensorboard
- *iterations* : Number of training iterations
- *batch_size* : Number of elements (scenes) in a training mini-batch
- *test_batch_size* : Number of elements (scenes) in a testing mini-batch
- *dataset* : Dataset to train on, used to find dataset if no directory is provided
- *cached_dataset* : Enables loading pre-rendered datasets
- *samples_per_pixel* : Number of samples to use while path tracing data
- *views_per_scene* : Number of observations and query views (N+1)
- *random_num_views* : Randomize the number of observations used per mini-batch [1, *views_per_scene* - 1]
- *latent_separation* : Enables partitioning into lighting, geometry and material partitions
- *adaptive_separation* : Enables adaptively learning partition sizes
- *empty_partition* : Use the null partition during training
- *partition_loss* : Weight of the null partition loss (beta)

### Model options

- *pose_dim* : The size of the camera coordinate inputs, should be 16 unless code changed
- *output_pass* : Defines which output buffer is our goal, typically the 'beauty' buffer
- *representations* : List of encoder networks
- *generators* : List of generator networks

#### Representation options

- *type* : Encoder architecture, either 'Pool' or 'Pyramid'
- *detach* : Enables using a pre-trained encoder
- *aggregation_func* : How to aggregate the representations for each observation, either 'Mean' or 'Max'
- *observation_passes* : Buffers provided to the encoder with each observation
- *representation_dim* : Dimensionality of the final neural scene representation
- *render_size* : The image size of the observations
- *start_sigmoid* : Initial smoothness of partition boundaries
- *end_sigmoid* : Smoothness of partition boundaries once network has been trained *sharpening* iterations
- *sharpen_sigmoid* : Smoothness of partition boundaries once network has been trained for *sharpening* + *final_sharpening* iterations
- *sharpening* : Number of iterations to blend between *start_sigmoid* and *end_sigmoid*
- *final_sharpening* : Number of iterations to blend between *end_sigmoid* and *sharpen_sigmoid*
- *gradient_reg* : Weight of gradient regularization (Kulkarni et. al)

#### Generator options

- *query_passes* : Buffers to provide to generator as input
- *output_passes* : Buffers the generator outputs
- *render_size* : Expected input and output image size
- *representations* : Indices of encoder networks used to construct input scene representations (needs to be changed if using multiple representation networks)
- *override_sample* : The outputs of this network are used to replace the ground-truth buffers during testing, can be used to forward outputs of one generator to another
- *override_train* : The outputs of this network are used to replace the ground-truth buffers during training, can be used to forward outputs of one generator to another
- *discard_loss* : The loss of this network is disabled
- *detach* : Gradients are not propagated through the network
- *type* : Generator architecture to use
    - **GQN**:
        - *cell_type* : 'GRU' or 'LSTM'
        - *latent_dim* : Depth of latent variables 'z'
        - *state_dim* : Depth of recurrent state
        - *core_count* : Number of recurrent cells
        - *downscaling* : Scaling down factor from *render_size*
        - *weight_sharing* : Share weights for the recurrent cells
        - *upscale_sharing* : Share weights of deconvolution scaling from *render_size* / *downscaling* to *render_size*
    - **Unet**:
        - *loss* : Loss function to use (default: ['logl1', 'logssim'])
        - *end_relu* : Applies a final ReLU to make sure images are non-negative
    - **PixelCNN** (Pixel Generator):
        - *loss* : Loss function to use (default: ['logl1', 'logssim'])
        - *end_relu* : Applies a final ReLU to make sure images are non-negative
        - *layers* : Number of hidden layers
        - *hidden_units* : Number of hidden units in each layer
        - *propagate_buffers* : Input buffers again at every layer
        - *propagate_representation* : Input representation again at every layer
        - *propagate_viewpoint* : Input the camera coordinates again at every layer
    - **Sum** (Helper that sums all *query_passes* and creates the output, can be used to sum direct and indirect outputs) 
    - **Copy** (Helper that can reuse a network):
        - *index* : Index of generator to reuse

### Available buffers

- *beauty* : Color image
- *position* : World position buffer
- *normal* : Normal buffer
- *depth* : Depth buffer
- *id* : Object id
- *albedo* : Albedo buffer
- *roughness* : Roughness buffer (in our material model roughness controls both specular coefficient and roughness)
- *direct* : Direct lighting buffer
- *indirect* : Indirect lighting buffer (*beauty* - *direct*)
- *diffuse* : Diffuse shading buffer
- *specular* : Specular shading buffer (*beauty* - *diffuse*)
- *mirror* : Perfect reflection vector
- *mirror_hit* : Intersection position of ray fired in *mirror* direction
- *mirror_normal* : Intersection normal of ray fired in *mirror* direction
- *shadows* : Direct shadowing buffer
- *ao* : Ambient occlusion buffer


