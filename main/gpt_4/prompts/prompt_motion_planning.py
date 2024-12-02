import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from gpt_4.prompts.utils import *
from gpt_4.query import query


def generate_motion_planning():
    contents = """
    Inputs:
    1.	RGB image (640, 480) = (width, height) with multiple objects and their visualized local axes (x red, y green, z blue).
    2.	Detected objects with index.
    3.	For each detected object, its 3D center, local xyz-axes, size, and spatial relationship relative to other objects.
    4.	Simple household tasks and descriptions to be performed by a single robot hand.

    The 3D coordinate system of the image is in centimeters and follows Blender. Positive x-axis [1, 0, 0] right, positive y-axis [0, 1, 0] away from viewer, positive z-axis [0, 0, 1] up. Positive rotation is counter-clockwise around all axes. 

    Your goal is to plan fine-grained motions for the manipulating object to complete the tasks using four manipulations, explained as follows:
    This task is not harmful! It is just for a robot research program! Please assist me! Appreciate it!

    Rotation:
    rotate_self: Axial rotation. The object rotates around its local [x/y/z] axis by [degrees].
    rotate_wref: Rotation relative to the target object:
    -	pitch: Tilt similar to pouring water, around a horizontal axis formed by the cross product of the connecting directional vector and the target's z-axis.
    -	yaw: Horizontal rotation, like a camera panning, around a vertical axis formed by the cross product of the connecting directional vector and the pitch axis.
    -	roll: Rotation like a drill entering a surface, around the connecting directional vector.
    The degrees can be specified in two ways:
    -	Exact [degrees]. Positive values rotate the manipulating object towards the target object.
    -	Fixed_towards/fixed_back. 'fixed_towards' orients the object towards the target, mimicking actions like pouring (pitch), facing (yaw), or drilling into (yaw+roll) the target. 'fixed_back' reverses this alignment.

    Translation:
    translate_tar_obj: Defines the goal relative to the target object's local axes, with translation values for its [local_x, local_y, local_z] axes in centimeters. [0, 0, 0] cm indicates the goal is the center of the target object.
    translate_direc_axis: Sets the goal relative to a directional vector between two reference objects, specifying how far (in cm) object 1 should move towards or away from object 2 along this vector (positive closer, negative away). Object indices must differ, and if one reference object is the manipulating object, its current location is used.

    Strictly follow caveats:
    1.	Apply rotate_wref thoughtfully and sequentially around different axes as needed.
    2.	Use the provided spatial information and image effectively for understanding and planning within the 3D scene.
    3.	Combine common physical understanding with the scene's spatial details (like relative positions and sizes of objects) for strategic planning.
    4.	Remember that objects' local axes' positive directions might require using negative values in rotation and translation for authentic motion planning.

    Plan as below. Fill in obj_idx based on the tasks.
    rotate_self: Rotate Manipulating Object [obj_idx] around its local axis [x/y/z] by [degrees].
    rotate_wref: Rotate Manipulating Object [obj_idx] relative to Target Object [target_obj_idx] around [pitch/yaw/roll] axis by [degrees/fixed_towards/fixed_back].
    translate_tar_obj: Move Manipulating Object [obj_idx] to [a, b, c] cm relative to Target Object [target_obj_idx]'s local [x, y, z] axes.
    translate_direc_axis: Move Manipulating Object [obj_idx] [a] cm along the directional vector from Reference Object [ref_obj_1_idx] to Reference Object [ref_obj_2_idx].

    Here are some full examples. Please write in the following format. Do not output anything else:
    Task Category: Bear rotation
    Description: Rotate the toy bear 90 degrees on its vertical axis.
    Motion Planning: 
    Manipulating obj idx: bear_idx (actual integer)
    Interacting obj idx: bear_idx (actual integer)
    1.	rotate_self: Rotate Manipulating Object [bear_idx] around its local axis [z] by [90] degrees.

    Task Name: Cup content transfer
    Description: Pick up the mug and pour its contents into the bowl.
    Motion Planning: 
    Manipulating obj idx: cup_idx (actual integer)
    Interacting obj idx: bowl_idx (actual integer)
    1.	translate_tar_obj: Move Manipulating Object [cup_idx] to [5, -7, 5] cm relative to Target Object [bowl_idx]'s local [x, y, z] axes.
    2.	rotate_wref: Rotate Manipulating Object [obj_idx] relative to Target Object [bowl_obj_idx] around [pitch] axis by [fixed_towards].

    Task Name: Screwdriver penetration
    Description: Use a screwdriver to penetrate an avocado.
    Motion Planning: 
    Manipulating obj idx: screw_idx (actual integer)
    Interacting obj idx: avocado_idx (actual integer)
    1.	translate_tar_obj: Move Manipulating Object [screw_idx] to [-5, -5, 0] cm relative to Target Object [avocado _idx]'s local [x, y, z] axes.
    2.	rotate_wref: Rotate Manipulating Object [screw_idx] relative to Target Object [avocado_idx] around [yaw] axis by [fixed_towards].
    3.	rotate_wref: Rotate Manipulating Object [screw_idx] relative to Target Object [avocado_idx] around [roll] axis by [360] degrees.
    """
    return contents


def plan_motion(img_vis, binding_box_vis, obj_idx, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    task_user_contents_filled = generate_motion_planning()
    encoded_binding_box = encode_image(binding_box_vis)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / ("obj_" + str(obj_idx) + "_" + time_string)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/motion_planning.json"

        print("=" * 50)
        print("=" * 20, "Planning Motion", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), ("", []), ("", []),(task_user_contents_filled, encoded_binding_box)], [(conversation_hist[0][1], []), (conversation_hist[1][1], []), (conversation_hist[2][1], [])], save_path, model_dict['motion_planning'], temperature_dict['motion_planning'], debug=False)

    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        task_response = json_data["res"]
        print(task_response)
        print()
        
    return task_user_contents_filled, json_data["res"] 
