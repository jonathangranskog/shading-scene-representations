import numpy as np
import random
import json
import pyrr
import math
import renderer.randomize.object as obj
import renderer.randomize.material as mat
import renderer.randomize.camera as cam
import colorsys as clr
import pyrr.matrix44 as m44

def select_randomizer(scene, seed):
    if scene == "room":
        return RoomRandomizer(seed)
    elif scene == "manylightroom":
        return ManyLightRoomRandomizer(seed)
    elif scene == "manymaterialroom":
        return ManyMaterialRoomRandomizer(seed)
    elif scene == "manymanyroom":
        return ManyManyRoomRandomizer(seed)
    elif scene == "fewmaterialroom":
        return FewMaterialRoomRandomizer(seed)
    else:
        return ArchvizRandomizer(seed)


# Base class for a generic scene randomizer
# actual scene types need to inherit this to construct JSON files
# that can be read by the scene renderer
class SceneRandomizer():
    def __init__(self, seed):
        if seed == -1:
            seed = random.randint(1, 100000000)
        self.seed = seed
        self.camera = None
        self.geometry = {}
        self.lights = []
        self.materials = {}
        self.bounces = 8
        self.cache = True
        self.hide_lights = False
        self.top_level_acceleration = "NoAccel"

    def random_scene(self):
        self.randomize_lighting()
        self.randomize_geometry()
        self.randomize_materials()

    def random_view(self):
        pass

    def randomize_lighting(self):
        pass

    def randomize_materials(self):
        pass

    def randomize_geometry(self):
        pass

    def animate_view(self, t):
        pass
    
    def generate_params(self):
        # This function converts all the class variables into a single dictionary
        # such that it can be saved as a json file
        params = {}
        params["seed"] = self.seed
        params["cache"] = self.cache
        params["bounces"] = self.bounces
        params["hide_lights"] = self.hide_lights
        params["top_level_acceleration"] = self.top_level_acceleration

        # Save camera
        cam = {}
        cam["position"] = list(self.camera.position)
        cam["lookat"] = list(self.camera.lookat)
        cam["fov"] = self.camera.fov
        cam["near"] = self.camera.near
        cam["far"] = self.camera.far
        params["camera"] = cam
        
        # Save materials and convert all numpy arrays to lists
        materials = self.materials.copy()
        for key in self.materials:
            materials[key] = self.materials[key].as_dict()
        params["materials"] = materials

        # Save light sources
        lights = []
        for light in self.lights:
            if light.visible:
                lights.append(light.as_dict())
        params["lights"] = lights

        # Save geometry by unpacking groups
        geometry = []
        for group in self.geometry:
            for obj in self.geometry[group]:
                if obj.visible:
                    geometry.append(obj.as_dict())
        params["geometry"] = geometry
             
        return params

    def get_json(self, params):
        return json.dumps(params)

    def save_json(self, filename, params):
        with open(filename, 'w') as scene_file:
            json.dump(params, scene_file, indent=4)

'''
The room scene is our playground scene -- it resembles a Cornell box scene with brightly colored walls and a few primitive objects.

The scene objects are randomly selected from a set of five objects; a cube, a sphere, a tetrahedron, a dodecahedron and a Utah teapot. 
The materials of these objects are randomized to be either diffuse, glossy specular and ideal mirror. These objects are randomly placed
in the scene such that they do not intersect each other. Everything is illuminated by a single spherical light source with constant intensity. 
'''
class RoomRandomizer(SceneRandomizer):
    def __init__(self, seed):
        super(RoomRandomizer, self).__init__(seed)
        self.bounces = 4
        self.cache = True
        self.hide_lights = True
        self.top_level_acceleration = "NoAccel"

        # Add materials
        self.materials["ground"] = mat.Material(id=0)
        self.materials["ceiling"] = mat.Material(id=1)
        self.materials["back_wall"] = mat.Material(id=2)
        self.materials["left_wall"] = mat.Material(id=3)
        self.materials["right_wall"] = mat.Material(id=4)
        self.materials["front_wall"] = mat.Material(id=5)

        self.materials["sphere"] = mat.Material(id=10)
        self.materials["cube"] = mat.Material(id=9)
        self.materials["dodecahedron"] = mat.Material(id=7)
        self.materials["cylinder"] = mat.Material(id=8)
        self.materials["teapot"] = mat.Material(id=11)

        # Add scene objects
        self.geometry["objs"] = []
        self.geometry["objs"].append(obj.Sphere(np.array([0.0, 0.5, 0.0]), 0.5, material="sphere"))
        self.geometry["objs"].append(obj.File("renderer/data/cube_bigger.obj", material="cube"))
        self.geometry["objs"].append(obj.File("renderer/data/dodecahedron.obj", material="dodecahedron"))
        self.geometry["objs"].append(obj.File("renderer/data/cylinder.obj", material="cylinder"))
        self.geometry["objs"].append(obj.File("renderer/data/teapot2.obj", material="teapot"))

        self.geometry["walls"] = []
        self.geometry["walls"].append(obj.Grid(position=np.array([0, 1.0, -3]), normal=np.array([0, 0, 1]), size=np.array([6, 3, 1]), material="back_wall"))
        self.geometry["walls"].append(obj.Grid(position=np.array([0, 1.0, 3]), normal=np.array([0, 0, -1]), size=np.array([6, 3, 1]), material="front_wall"))
        self.geometry["walls"].append(obj.Grid(position=np.array([-3, 1.0, 0]), normal=np.array([1, 0, 0]), size=np.array([1, 3, 6]), material="left_wall"))
        self.geometry["walls"].append(obj.Grid(position=np.array([3, 1.0, 0]), normal=np.array([-1, 0, 0]), size=np.array([1, 3, 6]), material="right_wall"))

        self.geometry["roof_and_ceil"] = []
        self.geometry["roof_and_ceil"].append(obj.Grid(position=np.array([0, -0.5, 0]), normal=np.array([0, 1, 0]), size=np.array([6, 1, 6]), material="ground"))
        self.geometry["roof_and_ceil"].append(obj.Grid(position=np.array([0, 2.5, 0]), normal=np.array([0, -1, 0]), size=np.array([6, 1, 6]), material="ceiling"))

        # Add light object
        self.materials["light"] = mat.Material(id=6, color=np.array([0, 0, 0]), emission=np.array([35, 35, 35]))
        self.lights.append(obj.Sphere(np.array([0.0, 0.5, 0.0]), 0.125, material="light"))

        # Add scene camera
        self.camera = cam.Camera(np.array([1, 1, 1]), np.array([0, 0, 0]), 75.0, 0.01, 100.0)

    def random_view(self):
        x = math.sin(random.random() * 2 * math.pi) * (1.5 + 1.5 * random.random())
        z = math.cos(random.random() * 2 * math.pi) * (1.5 + 1.5 * random.random())
        pos = np.array([x, random.random() + 1.0, z])
        target = np.array([random.random(), random.random() * 2, random.random()])

        self.camera.position = pos
        self.camera.lookat = target

    def animate_view(self, t):
        x = math.sin(t * 2 * math.pi) * 2.5
        z = math.cos(t * 2 * math.pi) * 2.5
        pos = np.array([x, 1.5 + 0.5 * math.cos(t * 2 * math.pi), z])
        target = np.array([0.0, 0.0, 0.0])

        self.camera.position = pos
        self.camera.lookat = target

    def randomize_lighting(self):
        position = np.array([random.random() * 4.5 - 2.25, random.random() * 0.85 + 1., random.random() * 4.5 - 2.25])
        transform = m44.create_from_translation(position)
        self.lights[0].transform = transform

    def randomize_materials(self):
        for wall in self.geometry["walls"]:
            self.materials[wall.material].color = np.asarray(clr.hsv_to_rgb(random.random(), 1.0, 1.0))

        for obj in self.geometry["objs"]:
            self.materials[obj.material].color = np.asarray(clr.hsv_to_rgb(random.random(), 1.0, 1.0))
            
            spec = random.random()
            if spec < 0.33:
                roughness = 1.
            elif spec < 0.66:
                roughness = 0.5
            else:
                roughness = 0.05
            self.materials[obj.material].roughness = roughness

    def randomize_geometry(self):
        indices = list(range(len(self.geometry["objs"])))
        random.shuffle(indices)

        taken_positions = []

        # Place objects randomly in the scene while avoiding intersections
        for l in range(len(self.geometry["objs"])):
            index = indices[l]
            visibility = random.random() > 0.5
            position = np.array([0, 0])
            if visibility:
                occupied = False
                k = 0
                while k < 5:
                    occupied = False
                    position[0] = random.random() * 4.5 - 2.25
                    position[1] = random.random() * 4.5 - 2.25

                    # See if there are any intersections
                    for taken in taken_positions:
                        length = np.linalg.norm(position - taken)
                        if length < 1.55:
                            occupied = True
                            break
                    
                    k += 1

                    # Found an empty position
                    if not occupied:
                        break

                if not occupied:
                    taken_positions.append(position*1)
                else:
                    visibility = False
            
            if visibility:
                translation = m44.create_from_translation(np.array([position[0], -0.5, position[1]]))
                rotation = m44.create_from_y_rotation(2 * math.pi * random.random())
                transform = m44.multiply(rotation, translation)
                self.geometry["objs"][index].visible = True
            else:
                transform = m44.create_from_translation(np.array([0, -5, 0]))
                self.geometry["objs"][index].visible = False

            self.geometry["objs"][index].transform = transform

class ManyLightRoomRandomizer(RoomRandomizer):
    def __init__(self, seed):
        super(ManyLightRoomRandomizer, self).__init__(seed)

        self.lights = []
        
        # Add light object
        num_lights = 10
        for i in range(num_lights):
            self.materials["light" + str(i)] = mat.Material(color=np.array([0, 0, 0]), emission=np.array([10, 10, 10]))
            self.lights.append(obj.Sphere(np.array([0.0, 0.5, 0.0]), 0.125, material="light" + str(i)))
        
    def randomize_lighting(self):
        for i in range(len(self.lights)):
            if random.random() < 0.5:
                self.lights[i].visible = False
            else:
                self.lights[i].visible = True
            position = np.array([random.random() * 4.5 - 2.25, random.random() * 0.85 + 1., random.random() * 4.5 - 2.25])
            self.lights[i].transform = m44.create_from_translation(position)
            self.materials["light" + str(i)].emission = np.array([15, 15, 15]) * (random.random() * 0.9 + 0.1)

class ManyMaterialRoomRandomizer(RoomRandomizer):
    def __init__(self, seed):
        super(ManyMaterialRoomRandomizer, self).__init__(seed)

    def randomize_materials(self):
        for wall in self.geometry["walls"]:
            self.materials[wall.material].color = np.array([random.random(), random.random(), random.random()])

        for obj in self.geometry["objs"]:
            self.materials[obj.material].color = np.array([random.random(), random.random(), random.random()])
            self.materials[obj.material].roughness = random.random() * 0.95 + 0.05

class ManyManyRoomRandomizer(ManyLightRoomRandomizer):
    def __init__(self, seed):
        super(ManyManyRoomRandomizer, self).__init__(seed)

    def randomize_materials(self):
        for wall in self.geometry["walls"]:
            self.materials[wall.material].color = np.array([random.random(), random.random(), random.random()])

        for obj in self.geometry["objs"]:
            self.materials[obj.material].color = np.array([random.random(), random.random(), random.random()])
            self.materials[obj.material].roughness = random.random() * 0.95 + 0.05

class FewMaterialRoomRandomizer(ManyLightRoomRandomizer):
    def __init__(self, seed):
        super(FewMaterialRoomRandomizer, self).__init__(seed)

    def randomize_lighting(self):
        for i in range(len(self.lights)):
            if random.random() < 0.5:
                self.lights[i].visible = False
            else:
                self.lights[i].visible = True
            position = np.array([random.random() * 4.5 - 2.25, random.random() * 0.85 + 1., random.random() * 4.5 - 2.25])
            self.lights[i].transform = m44.create_from_translation(position)
            self.materials["light" + str(i)].emission = np.array([random.random() * 15, random.random() * 15, random.random() * 15])
    
    def randomize_materials(self):
        for wall in self.geometry["walls"]:
            self.materials[wall.material].color = np.asarray(clr.hsv_to_rgb(random.random(), 1.0, 1.0))

        for obj in self.geometry["objs"]:
            self.materials[obj.material].color = np.array([1.0, 1.0, 1.0])
            self.materials[obj.material].roughness = 1.0

'''
For the Archviz scene, we envision a future where a neural renderer is specialized on rendering scenes for 
architectural visualization of interior scenes, i.e. the renderer can efficiently render these types of scenes
without the need for ray tracing.

This scene consists of a few objects set within an interior environment. The scene is separated into two areas,
one living room area and one for dining. The dining area consists of a dining table with up to four chairs whereas
the living room contains a sofa, an armchair, a coffee table and a carpet. The walls are all textured with the same
repeating texture selected from a set of four textures. The scene is illuminated by two light sources; a large area light
that is supposed to resemble a large window and a ceiling light that moves around randomly. Additionally, a teapot is randomly
placed in the scene and a mirror hangs on the back wall of the room. 
'''
class ArchvizRandomizer(SceneRandomizer):
    def __init__(self, seed):
        super(ArchvizRandomizer, self).__init__(seed)
        self.bounces = 8
        self.cache = True
        self.hide_lights = False
        self.top_level_acceleration = "NoAccel"

        # Add materials
        self.materials["ground"] = mat.Material(id=0)
        self.materials["ceiling"] = mat.Material(id=1)
        self.materials["back_wall"] = mat.Material(id=2)
        self.materials["left_wall"] = mat.Material(id=3)
        self.materials["right_wall"] = mat.Material(id=4)
        self.materials["front_wall"] = mat.Material(id=5)

        self.materials["table"] = mat.Material(id=9)
        self.materials["chair"] = mat.Material(id=8)
        self.materials["sofa"] = mat.Material(id=10)
        self.materials["armchair"] = mat.Material(id=11)
        self.materials["rug"] = mat.Material(id=12)
        self.materials["lamp"] = mat.Material(id=13, color=np.array([0.33, 0.33, 1.]), roughness=0.5)
        self.materials["coffee_table"] = mat.Material(id=14)
        self.materials["teapot"] = mat.Material(id=15)
        self.materials["frame"] = mat.Material(id=16)
        self.materials["mirror"] = mat.Material(id=17, roughness=0.05)
        
        # Add scene objects
        self.geometry["tables"] = []
        self.geometry["chairs"] = []
        self.geometry["sofas"] = []
        self.geometry["coffee_tables"] = []
        self.geometry["rug"] = []
        self.geometry["armchair"] = []
        self.geometry["walls"] = []
        self.geometry["floor_ceil"] = []
        self.geometry["teapot"] = []
        self.geometry["mirror"] = []
        self.geometry["lamp"] = []
        self.geometry["window_light"] = []

        self.geometry["tables"].append(obj.File("renderer/data/table2.obj", material="table"))
        self.geometry["tables"].append(obj.File("renderer/data/table3.obj", material="table"))
        self.geometry["tables"].append(obj.File("renderer/data/table4.obj", material="table"))

        chair_models = ["renderer/data/chair2.obj", "renderer/data/chair3.obj", "renderer/data/chair4.obj"]
        self.num_chair_models = len(chair_models)
        for i in range(len(chair_models)):
            for j in range(4):
                self.geometry["chairs"].append(obj.File(chair_models[i], material="chair"))
                self.geometry["chairs"].append(obj.File(chair_models[i], material="chair"))
                self.geometry["chairs"].append(obj.File(chair_models[i], material="chair"))
                self.geometry["chairs"].append(obj.File(chair_models[i], material="chair"))
        
        self.geometry["sofas"].append(obj.File("renderer/data/sofa.obj", material="sofa"))
        self.geometry["sofas"].append(obj.File("renderer/data/sofa2.obj", material="sofa"))

        self.geometry["armchair"].append(obj.File("renderer/data/armchair.obj", material="armchair"))
        self.geometry["rug"].append(obj.File("renderer/data/rug.obj", material="rug"))
        self.geometry["lamp"].append(obj.File("renderer/data/lamp.obj", material="lamp"))
        self.geometry["teapot"].append(obj.File("renderer/data/teapot3.obj", material="teapot"))
        self.geometry["mirror"].append(obj.File("renderer/data/frame.obj", material="frame"))
        self.geometry["mirror"].append(obj.File("renderer/data/canvas.obj", material="mirror"))

        self.geometry["coffee_tables"].append(obj.File("renderer/data/coffee_table.obj", material="coffee_table"))
        self.geometry["coffee_tables"].append(obj.File("renderer/data/coffee_table2.obj", material="coffee_table"))

        self.geometry["walls"].append(obj.Grid(position=np.array([0, 1.3, -3.25]), normal=np.array([0, 0, 1]), size=np.array([7, 2.6, 2.6], dtype='float'), material='back_wall'))
        self.geometry["walls"].append(obj.Grid(position=np.array([-3.5, 1.3, 0]), normal=np.array([1, 0, 0]), size=np.array([2.6, 2.6, 6.5], dtype='float'), material='left_wall'))
        self.geometry["walls"].append(obj.Grid(position=np.array([3.5, 1.3, 0]), normal=np.array([-1, 0, 0]), size=np.array([2.6, 2.6, 6.5], dtype='float'), material='right_wall'))
        self.geometry["walls"].append(obj.Grid(position=np.array([0, 1.3, 3.25]), normal=np.array([0, 0, -1]), size=np.array([7, 2.6, 2.6], dtype='float'), material='front_wall'))

        self.geometry["floor_ceil"].append(obj.Grid(position=np.array([0, 0, 0]), normal=np.array([0, 1, 0]), size=np.array([7.1, 1, 6.6]), material='ground'))
        self.geometry["floor_ceil"].append(obj.Grid(position=np.array([0, 2.6, 0]), normal=np.array([0, -1, 0]), size=np.array([7.1, 1, 6.6]), material='ceiling'))
        
        # Add lights
        self.materials["wind_light"] = mat.Material(color=np.zeros(3), emission=np.array([0.5, 0.5, 0.5]), id=6)
        self.materials["ceil_light"] = mat.Material(color=np.zeros(3), emission=np.array([0.1, 0.1, 0.1]), id=7)
        self.lights.append(obj.Grid(position=np.array([0, 0, 0]), normal=np.array([0, -1, 0]), size=np.array([0.2, 0, 0.2]), material="ceil_light"))
        self.lights.append(obj.Grid(position=np.array([2, 1.3, 3.2499]), normal=np.array([0, 0, -1]), size=np.array([2, 2, 0]), material='wind_light'))

        # Add camera
        self.camera = cam.Camera(np.array([2, 1, 2]), np.array([0, 0, 0]), 45.0, 0.1, 10.0)

        # Add textures
        self.textures = []
        self.textures.append("renderer/data/wallpaper1.hdr")
        self.textures.append("renderer/data/wallpaper2.hdr")
        self.textures.append("renderer/data/wallpaper3.hdr")
        self.textures.append("renderer/data/wallpaper4.hdr")
        
        self.hide_transform = m44.create_from_translation(np.array([0, -3, 0]))

    def random_view(self):
        x = (random.random() - 0.5) * 2 * 1.25 - 1.75
        z = (random.random() - 0.5) * 2 * 2.5
        y = 1.6 + 0.2 * 2 * (random.random() - 0.5)
        pos = np.array([x, y, z])
        lookat = np.array([2.0 + 2 * (random.random() - 0.5), 1.2, 2.25 * (random.random() - 0.5)])
        
        self.camera.position = pos
        self.camera.lookat = lookat

    def animate_view(self, t):
        px = math.sin(t * 2 * math.pi) * 1.25 - 1.75
        py = 1.6 + math.sin(t * 2 * math.pi) * 0.2
        pz = math.cos(t * 2 * math.pi) * 2.5

        tx = 2.0 + math.cos(t * 2 * math.pi + math.pi/3)
        ty = 1.2
        tz = 1.125 * math.sin(t * 2 * math.pi + math.pi / 3)

        pos = np.array([px, py, pz])
        lookat = np.array([tx, ty, tz])
        self.camera.position = pos
        self.camera.lookat = lookat

    def iq_color(self, t, a, b, c, d):
        color = a + b * np.cos(2 * np.pi * (c * t + d))
        color = np.clip(color, 0, 1)
        return color

    def randomize_lighting(self):
        # Set light colors
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        c = np.array([1, 1, 1])
        d = np.array([0.5, 0.35, 0.2])

        color = self.iq_color(random.random(), a, b, c, d)
        emission = 15 + 8.5 * color
        self.materials[self.lights[1].material].emission = emission

        color = self.iq_color(random.random(), a, b, c, d)
        emission = 10 + 6 * color
        self.materials[self.lights[0].material].emission = emission

        # Move light source
        pos = np.array([1.6 + (random.random() - 0.5), 4.5 * (random.random() - 0.5)])
        self.lights[0].transform = m44.create_from_translation(np.array([pos[0], 1.8375, pos[1]]))
        self.geometry["lamp"][0].transform = m44.create_from_translation(np.array([pos[0], 1.835, pos[1]]))

    def randomize_materials(self):
        self.randomize_wall_materials()
        self.randomize_dining_materials()
        self.randomize_living_materials()
        self.randomize_teapot_material()

    def randomize_geometry(self):
        r = random.random() < 0.5
        self.randomize_dining_geometry(r)
        self.randomize_living_geometry(r)
        self.randomize_teapot_geometry(r)
        self.randomize_mirror_geometry()

    def randomize_wall_materials(self):
        wallpaperIndex = int(random.random() * len(self.textures))
        # Set frequencies
        if wallpaperIndex < 2:
            freq1 = np.array([2.6, 7])
            freq2 = np.array([2.6, 6.5])
        else:
            freq1 = np.ones(2)
            freq2 = np.ones(2)

        self.materials["back_wall"].texture_frequency = freq1
        self.materials["front_wall"].texture_frequency = freq1
        self.materials["right_wall"].texture_frequency = freq2
        self.materials["left_wall"].texture_frequency = freq2

        self.materials["back_wall"].texture = self.textures[wallpaperIndex]
        self.materials["front_wall"].texture = self.textures[wallpaperIndex]
        self.materials["right_wall"].texture = self.textures[wallpaperIndex]
        self.materials["left_wall"].texture = self.textures[wallpaperIndex]

        color = np.asanyarray(clr.hsv_to_rgb(random.random(), 1, 1))
        self.materials["back_wall"].color = color
        self.materials["front_wall"].color = color
        self.materials["right_wall"].color = color
        self.materials["left_wall"].color = color

        frameColor = random.random() * np.array([1.0, 1.0, 1.0])
        self.materials["frame"].color = frameColor
        self.materials["mirror"].roughness = 0.05

    def randomize_dining_materials(self):
        t = random.random()
        a = np.array([0.5, 0.4, 0.0])
        b = np.array([0.47, 0.5, 0.8])
        c = np.array([1.0, 1.0, 1.0])
        d = np.array([0.0, 0.0, 0.0])

        wood_color = self.iq_color(t, a, b, c, d)
        wood_roughness = (random.random() < 0.5) * 0.5 + 0.5
        self.materials["table"].color = wood_color
        self.materials["chair"].color = wood_color
        self.materials["table"].roughness = wood_roughness
        self.materials["chair"].roughness = wood_roughness
        
    def randomize_living_materials(self):
        t = random.random()
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        c = np.array([1.0, 1.0, 1.0])
        d = np.array([0.0, 0.1, 0.2])
        color = self.iq_color(t, a, b, c, d)
        self.materials["sofa"].color = color
        self.materials["armchair"].color = color

        t = random.random()
        color = self.iq_color(t, a, b, c, d)
        self.materials["rug"].color = color
        self.materials["coffee_table"].roughness = (random.random() < 0.5) * 0.5 + 0.5

    def randomize_teapot_material(self):
        r = random.random()
        if r < 0.33: roughness = 0.05
        elif r < 0.66: roughness = 0.5
        else: roughness = 1.0

        color = np.asanyarray(clr.hsv_to_rgb(random.random(), 1, 1))
        self.materials["teapot"].color = color
        self.materials["teapot"].roughness = roughness

    def randomize_dining_geometry(self, side):
        if side:
            area_pos = np.array([1.75 - 0.5 * random.random(), 0, 1.55])
        else:
            area_pos = np.array([1.75 - 0.5 * random.random(), 0, -1.75])

        if random.random() < 0.125:
            area_pos[1] = -3
            for table in self.geometry["tables"]:
                table.visible = False
            for chair in self.geometry["chairs"]:
                chair.visible = False

        self.dining_pos = area_pos
        dining_transform = m44.multiply(m44.create_from_y_rotation(random.random() * 2 * np.pi), m44.create_from_translation(area_pos))

        # Randomize table
        self.table_index = int(random.random() * len(self.geometry["tables"])) 
        for idx in range(len(self.geometry["tables"])):
            if idx == self.table_index:
                self.geometry["tables"][idx].transform = dining_transform
                self.geometry["tables"][idx].visible = True
            else:
                self.geometry["tables"][idx].transform = self.hide_transform
                self.geometry["tables"][idx].visible = False

        # Randomize chairs
        chairTranslations = np.array([[-0.45, 0, -0.75], [0.25, 0, -0.75], [-.45, 0, 0.75], [0.25, 0, 0.75]])
        chairTransformations = [m44.create_from_translation(x) for x in chairTranslations]
        chair_index = int(random.random() * self.num_chair_models)
        counter = 0
        for idx in range(len(self.geometry["chairs"])):
            if idx // self.num_chair_models == chair_index and random.random() > 0.25:
                matrix = chairTransformations[counter]
                if counter < 2:
                    matrix = m44.multiply(m44.create_from_y_rotation(np.pi), matrix)
                matrix = m44.multiply(m44.create_from_y_rotation((random.random() - 0.5) * np.pi / 6), matrix)
                matrix = m44.multiply(matrix, dining_transform)
                self.geometry["chairs"][idx].transform = matrix
                self.geometry["chairs"][idx].visible = True
                counter += 1
            else:
                self.geometry["chairs"][idx].transform = self.hide_transform
                self.geometry["chairs"][idx].visible = False
        
    def randomize_living_geometry(self, side):
        if side:
            area_pos = np.array([1.75 - 0.5 * random.random(), 0, -1.75])
        else:
            area_pos = np.array([1.75 - 0.5 * random.random(), 0, 1.25])

        if random.random() < 0.125:
            area_pos[1] = -3
            for sofa in self.geometry["sofas"]:
                sofa.visible = False
            self.geometry["armchair"][0].visible = False
            self.geometry["rug"][0].visible = False
            for coffee_table in self.geometry["coffee_tables"]:
                coffee_table.visible = False

        self.living_pos = area_pos
        living_transform = m44.multiply(m44.create_from_y_rotation(random.random() * 2 * np.pi), m44.create_from_translation(area_pos))
        
        # Select sofa and transform
        sofa_index = int(random.random() * len(self.geometry["sofas"]))
        sofa_transform = m44.multiply(m44.create_from_y_rotation(0.5 * np.pi), m44.create_from_translation(np.array([1.05, 0, 0.05])))
        for idx in range(len(self.geometry["sofas"])):
            if idx == sofa_index:
                self.geometry["sofas"][idx].transform = m44.multiply(sofa_transform, living_transform)
                self.geometry["sofas"][idx].visible = True
            else:
                self.geometry["sofas"][idx].transform = self.hide_transform
                self.geometry["sofas"][idx].visible = False
        
        armchair_transform = m44.multiply(m44.create_from_y_rotation(-0.83 * np.pi + 0.1 * np.pi * (random.random() - 0.5)), m44.create_from_translation(np.array([-0.9, 0, 0.65])))
        self.geometry["armchair"][0].transform = m44.multiply(armchair_transform, living_transform)
        self.geometry["armchair"][0].visible = True

        rug_transform = m44.multiply(m44.create_from_y_rotation(0.05 * np.pi * (random.random() - 0.5)), m44.create_from_translation(np.array([0, 0.005, 0])))
        self.geometry["rug"][0].transform = m44.multiply(rug_transform, living_transform)
        self.geometry["rug"][0].visible = True

        # Select coffee table and transform
        coffee_table_transform = m44.multiply(m44.create_from_y_rotation(-0.5 * np.pi), m44.create_from_translation(np.array([0.1, 0.006, 0])))
        coffee_table_index = int(random.random() * len(self.geometry["coffee_tables"]))
        for idx in range(len(self.geometry["coffee_tables"])):
            if idx == coffee_table_index:
                self.geometry["coffee_tables"][idx].transform = m44.multiply(coffee_table_transform, living_transform)
                self.geometry["coffee_tables"][idx].visible = True
            else:
                self.geometry["coffee_tables"][idx].transform = self.hide_transform
                self.geometry["coffee_tables"][idx].visible = False

    def randomize_teapot_geometry(self, side):
        r = random.random()
        # Put on living room table
        if r < 0.5:
            pos = np.array([0.1, 0.45, 0]) + self.living_pos
        # Put on dining room table
        else:
            if self.table_index == 0:
                pos = np.array([0, 0.7, 0]) + self.dining_pos
            elif self.table_index == 1 or self.table_index == 2:
                pos = np.array([0, 0.72, 0]) + self.dining_pos
            else:
                pos = np.array([0, 0.64, 0]) + self.dining_pos

        if pos[1] < 0:
            pos[1] = 0

        self.geometry["teapot"][0].transform = m44.multiply(m44.create_from_y_rotation(2 * np.pi * random.random()), m44.create_from_translation(pos))
            
    def randomize_mirror_geometry(self):
        pos = np.array([3.505, 1.4, (random.random() - 0.5) * 4])
        transform = m44.multiply(m44.create_from_y_rotation(0.5 * np.pi), m44.create_from_translation(pos))
        for obj in self.geometry["mirror"]:
            obj.transform = transform
            