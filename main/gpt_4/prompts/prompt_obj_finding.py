import numpy as np
import copy
import time, datetime
import os
import pathlib
import json
from gpt_4.prompts.utils import *
from gpt_4.query import query


def generate_obj_finding_contents():
    contents = """
    Input:
    RGB image (640, 480) = (width, height) with multiple objects.

    Your task is to identify objects by precise color, texture, and 2D spatial locations (in words). Do not use vague phrase like multi-colored.

    Please write in the following format. Do not output anything else:
    Object idx (actual integer, start from 0): x of color y, texture z at location w.
    """
    return contents 


def find_useful_object(img_vis, binding_box_vis, output_path, existing_response=None, temperature_dict=None, 
                  model_dict=None, conversation_hist=None):

    task_user_contents_filled = generate_obj_finding_contents()
    encoded_img = encode_image(img_vis)

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = output_path / time_string
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = str(save_folder) + "/obj_finding.json"

        print("=" * 50)
        print("=" * 20, "Finding Useful Objects", "=" * 20)
        print("=" * 50)
        
        json_data = query(system, [(task_user_contents_filled, encoded_img)], [], save_path, model_dict['obj_finding'], temperature_dict['obj_finding'], debug=False)
   
    else:
        with open(existing_response, 'r') as f:
            json_data = json.load(f)
        task_response = json_data["res"]
        print(task_response)
        print()
        
    return task_user_contents_filled, json_data["res"] 
