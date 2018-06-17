import bpy
from random import uniform

place_holder_img = "/home/long/data/digest_size/0001.png"
filename_file = "/home/long/dl/desmoker/train_filenames.txt"
datapath = "/home/long/data/daVinci/train/image_0/"
output_dir = "/home/long/data/tmp/"



def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines

def add_smoke():
    bpy.ops.object.modifier_add(type='SMOKE')
    bpy.ops.object.material_slot_add()
    bpy.ops.object.quick_smoke()
    bpy.data.objects["Cube"].scale = (0.3, 0.3, 0.3)
    bpy.data.objects["Cube"].location = (0.0, 0.0, 2.728)
    bpy.data.objects["Cube"].modifiers["Smoke"].flow_settings.smoke_color = (5, 5, 5)
    bpy.data.objects["Cube"].modifiers["Smoke"].flow_settings.density = 0.04
    bpy.data.objects["Cube"].modifiers["Smoke"].flow_settings.use_initial_velocity = True
    bpy.data.objects["Cube"].modifiers["Smoke"].flow_settings.velocity_factor = 1.5935
    bpy.data.objects["Cube"].modifiers["Smoke"].flow_settings.velocity_normal = 0.0465
    bpy.data.objects["Cube"].hide_render = True
    bpy.context.object.modifiers["Smoke"].domain_settings.use_adaptive_domain = True
    bpy.context.object.modifiers["Smoke"].domain_settings.use_high_resolution = True
    bpy.context.object.modifiers["Smoke"].domain_settings.alpha = 0.2474
    bpy.context.object.modifiers["Smoke"].domain_settings.beta = 0.6776
    bpy.context.object.modifiers["Smoke"].domain_settings.vorticity = 3.36
    
def set_camera():
    camera = bpy.data.objects["Camera"]
    camera.location = (0.0, 0.0, 6.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)

def random_camera():
    camera = bpy.data.objects["Camera"]
    camera.location = (uniform(-1.25, 1.25), uniform(-1.0, 1.0), uniform(4.0, 7.0))
    camera.rotation_euler = (0.0, 0.0, 0.0)
    
def add_background(filepath):
    img = bpy.data.images.load(filepath)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            space_data = area.spaces.active
            bg = space_data.background_images.new()
            bg.image = img
            space_data.show_background_images = True
            break

    texture = bpy.data.textures.new("Texture.001", 'IMAGE')
    texture.image = img
    bpy.data.worlds['World'].active_texture = texture
    bpy.context.scene.world.texture_slots[0].use_map_horizon = True
    bpy.context.scene.world.use_sky_paper = True
        
def start_render(fp):
        scn.render.filepath = fp
        bpy.ops.render.render(animation=True)
    
scn = bpy.context.scene
scn.frame_end=2
scn.render.resolution_x = 384
scn.render.resolution_y = 192
print("Hello Blender")
add_background(place_holder_img) # just a place holder
add_smoke()
set_camera()
bpy.ops.render.render(write_still=True)
filenames = read_text_lines(filename_file)
for filename in filenames:
    img = bpy.data.images.load(datapath+filename)
    bpy.data.textures["Texture.001"].image=img
    random_camera()
    start_render(output_dir+filename.split('.')[0]+"_#")

