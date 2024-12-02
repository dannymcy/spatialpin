import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from gpt_4.prompts.utils import *
from gpt_4.query import query


def generate_code_prompt():
    contents = """
    Inputs:
    1.	Simple household tasks and descriptions to be performed by a single robot hand.
    2.	Planned motion.

    Your goal is to generate obj_idx and motion_list for progamming.

    Detailed instructions:
        # obj_idx: Index of the manipulating object.
        # motion_list: List of motions, each a tuple defining a specific action:
        # - 'rotate_self': Axial rotation of the object around its own local axis.
        #   Tuple structure: ("rotate_self", angle_degrees, axis_type, axis_obj_idx)
        #   - angle_degrees: Rotation angle in degrees.
        #   - axis_type: Axis of obj_idx for rotation ('local_x', 'local_y', 'local_z').
        #   - axis_obj_idx: Always obj_idx, indicating self-rotation.
        #
        # - 'rotate_wref': Rotation of the object with reference to a target object.
        #   Tuple structure: ("rotate_wref", angle_degrees, axis_type, target_obj_idx)
        #   - angle_degrees: Rotation angle in degrees, or 'fixed_towards'/'fixed_back'.
        #   - axis_type: Axis for rotation ('pitch', 'yaw', 'roll').
        #   - target_obj_idx: Index of the reference object for rotation.
        #
        # - 'translate_tar_obj': Translation towards a goal relative to a target object.
        #   Tuple structure: ("translate_tar_obj", translation_vector, target_obj_idx)
        #   - translation_vector: Distances [dx, dy, dz] to move along each target axis.
        #   - target_obj_idx: Index of the target object for relative translation.
        #
        # - 'translate_direc_axis': Translation between two reference objects.
        #   Tuple structure: ("translate_direc_axis", distance, ref_obj_1_idx, ref_obj_2_idx)
        #   - distance: Distance in cm to travel from ref_obj_1_idx to ref_obj_2_idx.
        #   - ref_obj_1_idx: Index of the starting reference object.
        #   - ref_obj_2_idx: Index of the destination reference object.
        #
        # Sequential motions in motion_list are accumulative.
        # Examples of motion tuples:
        # - ("rotate_self", 90, "local_x", axis_obj_idx)
        # - ("rotate_wref", 90, "pitch", target_obj_idx)
        # - ("translate_tar_obj", [2, 1, 0], target_obj_idx)
        # - ("translate_direc_axis", 3, ref_obj_1_idx, ref_obj_2_idx)

    Here are some full examples. Please write in the following format. Do not output anything else:
    Task Name: Cup content transfer
    Description: Pick up the mug and pour its contents into the bowl.
    Motion Planning: 
    Manipulating obj idx: cup_idx (actual integer, 0)
    Interacting obj idx: bowl_idx (actual integer, 3)
    1.	translate_tar_obj: Move Manipulating Object [cup_idx] to [5, -7, 5] cm relative to Target Object [bowl_idx]'s local [x, y, z] axes.
    2.	rotate_wref: Rotate Manipulating Object [obj_idx] relative to Target Object [bowl_obj_idx] around [pitch] axis by [fixed_towards].
    obj_idx = 0
    motion_list = [("translate_tar_obj", [5, -7, 5], 3), ("rotate_wref ", 75, "pitch", 3)]
    """
    return contents


def generate_code(img_vis, binding_box_vis, obj_idx, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    task_user_contents_filled = generate_code_prompt()

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / ("obj_" + str(obj_idx) + "_" + time_string)
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/code_generation.json"

        print("=" * 50)
        print("=" * 20, "Generating Code", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [("", []), (task_user_contents_filled, [])], [(conversation_hist[-1][1], [])], save_path, model_dict['code_generation'], temperature_dict['code_generation'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        task_response = json_data["res"]
        print(task_response)
        print()
        
    return task_user_contents_filled, json_data["res"] 
