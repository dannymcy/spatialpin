import os
import json
import cv2
import io
import argparse
import pathlib
import copy
import time
import shutil
import requests

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from simnet.lib import camera
from simnet.lib.transform import Pose

from PIL import Image, ImageFilter
from lang_segment_anything.lang_sam import LangSAM
from lang_segment_anything.lang_sam.viz import *

from gpt_4.prompts.prompt_obj_finding import find_useful_object
from gpt_4.prompts.prompt_task_proposal import propose_task
from gpt_4.prompts.utils import load_response, extract_useful_object, extract_words_before, extract_code

from github import Github

from torchvision.transforms import Compose
from Depth_Anything.depth_anything.dpt import DepthAnything
from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from shutil import copy2
import pandas as pd


def depth_estimation(data_dir, output_dir, device_):
    encoder_list =['vits', 'vitb', 'vitl']
    encoder = encoder_list[2]

    margin_width = 50
    caption_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    DEVICE = device_

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        depth_output_path = img_full_path + '_depth_DA.png'

        raw_image = cv2.imread(color_path)
        if raw_image.shape[:2] != (480, 640):
            # Resize the image
            resized_image = cv2.resize(raw_image, (640, 480), interpolation=cv2.INTER_AREA)
        else:
            resized_image = raw_image

        # Convert color from BGR to RGB (cv2 uses BGR by default)
        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Save the image back to color_path
        cv2.imwrite(color_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        image = image / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint16)
        cv2.imwrite(depth_output_path, depth)


def obj_finding_lang_sam(data_dir, output_dir, temperature_dict, model_dict, start_over=False):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    obj_mask_output_dir = pathlib.Path(output_dir) / "obj_mask"
    binary_mask_output_dir = pathlib.Path(output_dir) / "binary_mask"
    centers_2d_output_dir = pathlib.Path(output_dir) / "centers_2d"
    obj_mask_output_dir.mkdir(parents=True, exist_ok=True)
    binary_mask_output_dir.mkdir(parents=True, exist_ok=True)
    centers_2d_output_dir.mkdir(parents=True, exist_ok=True)

    model = LangSAM()
    conversation_hist = []
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        img_vis = cv2.imread(color_path)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        img_vis_PIL = Image.open(color_path).convert("RGB")

        obj_finding_path = pathlib.Path(output_dir).parent / "gpt4_response" / "prompts/obj_finding" / str(i)
        if start_over:
            user, res = find_useful_object(img_vis, None, obj_finding_path, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
            time.sleep(20)
        else: 
            user, res = find_useful_object(img_vis, None, obj_finding_path, existing_response=load_response("obj_finding", obj_finding_path), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=None)
        conversation_hist.append([user, res])

        obj_text_list = extract_useful_object("obj_finding", obj_finding_path)
        masks_list = []
        for obj_text in obj_text_list:
            masks, boxes, phrases, logits = model.predict(img_vis_PIL, obj_text)
            if masks.shape[0] > 1:
                masks = masks[0]
            elif masks.numel() == 0:
                masks, boxes, phrases, logits = model.predict(img_vis_PIL, "object most similar to " + extract_words_before(obj_text, "of"))
    
            masks = masks.cpu().squeeze(0).numpy()
            masks_list.append(masks)
    
        save_mask_centers(masks_list, os.path.join(output_dir, "centers_2d", f"centers_2d_{i}.json"))
        save_separate_masks_transparent(img_vis, masks_list, os.path.join(output_dir, "obj_mask", f"masks_{i}"))
        save_separate_masks_binary(masks_list, os.path.join(output_dir, "binary_mask", f"masks_{i}"))
    
    return conversation_hist


def create_inpaint_masks(data_dir, output_dir):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    binary_mask_dir = pathlib.Path(output_dir) / "binary_mask"
    bg_mask_output_dir = pathlib.Path(output_dir) / "bg_mask"
    inpaint_mask_output_dir = pathlib.Path(output_dir) / "inpaint_mask"
    bg_mask_output_dir.mkdir(parents=True, exist_ok=True)
    inpaint_mask_output_dir.mkdir(parents=True, exist_ok=True)

    lama_inpainted_mask_output_dir = pathlib.Path(output_dir).parent / "lama" / "inpainted_mask"
    lama_inpainted_bg_mask_output_dir = pathlib.Path(output_dir).parent / "lama" / "inpainted_bg_mask"
    lama_inpainted_mask_output_dir.mkdir(parents=True, exist_ok=True)
    lama_inpainted_bg_mask_output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        img_vis = cv2.imread(color_path)

        file_count = sum(1 for f in binary_mask_dir.iterdir() if f.is_file() and f.name.startswith(f"masks_{i}_obj_"))
        for obj_idx in range(file_count):
            binary_mask_path = binary_mask_dir / f"masks_{i}_obj_{obj_idx}.png"
            inpaint_rgb_mask_path = inpaint_mask_output_dir / f"image{i}_obj_{obj_idx}.png"
            inpaint_binary_mask_path = inpaint_mask_output_dir / f"image{i}_obj_{obj_idx}_mask000.png"
            save_separate_masks_inpaint(i, img_vis, str(binary_mask_dir), str(binary_mask_path), str(inpaint_rgb_mask_path), str(inpaint_binary_mask_path), dilation_pixels=5)

        bg_rgb_mask_path = bg_mask_output_dir / f"image{i}.png"
        bg_binary_mask_path = bg_mask_output_dir / f"image{i}_mask000.png"
        save_separate_masks_inpaint(i, img_vis, str(binary_mask_dir), str(binary_mask_path), str(bg_rgb_mask_path), str(bg_binary_mask_path), background=True)


def inpainted_obj_finding_lang_sam(data_dir, output_dir):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    obj_mask_output_dir = pathlib.Path(output_dir) / "obj_mask"
    binary_mask_output_dir = pathlib.Path(output_dir) / "binary_mask"
    centers_2d_output_dir = pathlib.Path(output_dir) / "centers_2d"
    inpainted_mask_dir = pathlib.Path(output_dir).parent / "lama" / "inpainted_mask"

    model = LangSAM()
    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        obj_finding_path = pathlib.Path(output_dir).parent / "gpt4_response" / "prompts/obj_finding" / str(i)
        obj_text_list = extract_useful_object("obj_finding", obj_finding_path)
        masks_list, img_vis_list = [], []

        for obj_idx, obj_text in enumerate(obj_text_list):
            inpainted_rgb_mask_path = inpainted_mask_dir / f"image{i}_obj_{obj_idx}_mask000.png"
            img_vis = cv2.imread(str(inpainted_rgb_mask_path))
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            img_vis_list.append(img_vis)
            img_vis_PIL = Image.open(inpainted_rgb_mask_path).convert("RGB")
            
            masks, boxes, phrases, logits = model.predict(img_vis_PIL, obj_text)
            if masks.shape[0] > 1:
                masks = masks[0]
            elif masks.numel() == 0:
                masks, boxes, phrases, logits = model.predict(img_vis_PIL, "object most similar to " + extract_words_before(obj_text, "of"))

            # Special cases, an issue related to the way model.predict return masks
            if not (i == 28 and obj_idx == 1) and not (i == 36 and obj_idx == 1) and not (i == 45 and obj_idx == 1) and not (i == 46 and obj_idx == 2) and not (i == 48 and obj_idx == 3):
                masks = masks.cpu().squeeze(0).numpy()
            else:
                masks = cv2.imread(str(binary_mask_output_dir / f"masks_{i}_obj_{obj_idx}.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
            masks_list.append(masks)

        save_separate_masks_transparent(img_vis_list, masks_list, os.path.join(output_dir, "obj_mask", f"masks_{i}"))
        save_separate_masks_binary(masks_list, os.path.join(output_dir, "binary_mask", f"masks_{i}"))
        save_mask_centers(masks_list, os.path.join(output_dir, "centers_2d", f"centers_2d_{i}.json"))
 

def blur_background(data_dir, output_dir):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    lama_inpainted_bg_mask_output_dir = pathlib.Path(output_dir).parent / "lama" / "inpainted_bg_mask"

    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        lama_inpainted_bg_mask_output_path = lama_inpainted_bg_mask_output_dir / f"image{i}_mask000.png"
        lama_inpainted_bg_mask_blurred_output_path = lama_inpainted_bg_mask_output_dir / f"image{i}_mask000_blurred.png"
        background = Image.open(lama_inpainted_bg_mask_output_path)
        blurred_background = background.filter(ImageFilter.GaussianBlur(radius=2.5))
        blurred_background.save(lama_inpainted_bg_mask_blurred_output_path)


def task_proposal_gpt4(data_dir, output_dir, conversation_hist, temperature_dict, model_dict, start_over=False):
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    if conversation_hist is None:
        conversation_hist = []
        for i, img_path in enumerate(data_path):
            obj_finding_path = pathlib.Path(output_dir).parent / "gpt4_response" / "prompts/obj_finding" / str(i)
            subdirs = [d for d in os.listdir(obj_finding_path) if os.path.isdir(obj_finding_path / d)]
            subdirs.sort()
            latest_subdir = max(subdirs, key=lambda d: (obj_finding_path / d).stat().st_mtime)
            json_file_path = obj_finding_path / latest_subdir / "obj_finding.json"
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                user = json_data["user"]
                res = json_data["res"]
            conversation_hist.append([user, res])

    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png'
        img_vis = cv2.imread(color_path)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        obj_finding_path = pathlib.Path(output_dir).parent / "gpt4_response" / "prompts/obj_finding" / str(i)
        task_proposal_path = pathlib.Path(output_dir).parent / "gpt4_response" / "prompts/task_proposal" / str(i)
        obj_text_list = extract_useful_object("obj_finding", obj_finding_path)

        for obj_idx, _ in enumerate(obj_text_list):
          if start_over:
              propose_task(img_vis, None, obj_idx, task_proposal_path, existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=[conversation_hist[i]])
              time.sleep(20)
          else: 
              propose_task(img_vis, None, obj_idx, task_proposal_path, existing_response=load_response("task_proposal", task_proposal_path), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=[conversation_hist[i]])


# https://stackoverflow.com/questions/72668275/how-to-upload-an-image-file-to-github-using-pygithub
def git_push_image(git_token, git_repo, local_img_path, repo_img_path, commit_message, branch):
    g=Github(git_token)
    repo = g.get_repo(git_repo)

    img = cv2.imread(local_img_path, cv2.IMREAD_UNCHANGED)
    img_encode = cv2.imencode('.png', img)[1]
    img_data = io.BytesIO(img_encode).getvalue()

    try:
        # Try to get the existing file
        contents = repo.get_contents(repo_img_path, ref=branch)
        commit = repo.update_file(contents.path, commit_message, img_data, contents.sha, branch=branch)
    except:
        # If the file does not exist, create a new one
        commit = repo.create_file(repo_img_path, commit_message, img_data, branch)

    # Get the URL of the uploaded image
    file_info = commit['content']
    raw_url = file_info.download_url
    return raw_url


# https://docs.sudo.ai/
def generate_3d_model(data_dir, output_dir, git_token, git_repo, api_key):
    url = "https://platform.sudo.ai/api/v1/image-to-3d"
    request_headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    response_headers = {
        "x-api-key": api_key
    }
    data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    obj_mask_dir = pathlib.Path(output_dir) / "obj_mask"

    for i, img_path in enumerate(data_path):
        # if i != 0: continue  # you can control which image to run
        one2345_output_dir = pathlib.Path(output_dir).parent / "one2345_plus" / str(i)
        one2345_output_dir.mkdir(parents=True, exist_ok=True)
        file_count = sum(1 for f in obj_mask_dir.iterdir() if f.is_file() and f.name.startswith(f"masks_{i}_obj_"))
        
        for obj_idx in range(file_count):
            obj_mask_path = obj_mask_dir / f"masks_{i}_obj_{obj_idx}.png"
            image_url = git_push_image(git_token, git_repo, str(obj_mask_path), f"masks_{i}_obj_{obj_idx}.png", "", "master")
            data = {
                "image_url": image_url,
                "segm_mode": "auto"
            }

            # Step 1: Initiate 3D model generation 
            response = requests.post(url, json=data, headers=request_headers)  # this line costs One2345++ credits!!!
            if response.status_code == 200:
                # Step 2: Periodically check the status of the task
                response_json = response.json()
                task_id = response_json.get("id")
                meta_output_path = one2345_output_dir / f"{obj_idx}.json"
                with open(str(meta_output_path), 'w') as file:
                    json.dump({"task_id": task_id}, file)
                
                while True:
                    task_status_response = requests.get(f"https://platform.sudo.ai/api/v1/image-to-3d/{task_id}", headers=response_headers)
                    if task_status_response.status_code == 200:
                        task_status_data = task_status_response.json()

                        if task_status_data["status"] == "success":
                            # Process successful result
                            glb_url = task_status_data["models"]["glb"]
                            thumbnail_url = task_status_data["thumbnail"]
                            # Download and save the GLB file
                            one2345_output_path = one2345_output_dir / f"{obj_idx}.glb"
                            glb_response = requests.get(glb_url)
                            if glb_response.status_code == 200:
                                with open(str(one2345_output_path), 'wb') as f:
                                    f.write(glb_response.content)
                                with open(str(meta_output_path), 'w') as file:
                                    json.dump({"task_id": task_id, "glb_url": glb_url}, file)
                            else:
                                print(f"Error downloading GLB file: {glb_response.status_code}")
                            break
                        elif task_status_data["status"] == "failed":
                            print(f"Task {task_id} failed.")
                            break
                    
                    else:
                        print(f"Error checking task status: {task_status_response.status_code}")

                    time.sleep(5)  # Wait for some time before checking again
            else:
                print("Error initiating task:", response.status_code, response.text)


def create_shortlisted_folders(excel_path, original_base_dir, new_base_dir):
    # Assuming pandas is already imported and excel_path is defined
    df = pd.read_excel(excel_path)
    
    # Filter out rows where "Video Name" is NaN
    df = df.dropna(subset=["Video Name"])
    
    for i, row in df.iterrows():
        video_name = row["Video Name"]
        # Extract the unique identifier correctly
        unique_identifier = video_name.replace('video_', '').rsplit('.', 1)[0]
        
        # Assuming the first part of the unique identifier is the folder index
        folder_index = unique_identifier.split('_', 1)[0]
        
        original_folder_path = pathlib.Path(original_base_dir) / folder_index
        new_folder_path = pathlib.Path(new_base_dir) / folder_index
        os.makedirs(new_folder_path, exist_ok=True)  # Ensure the new folder exists
        
        # Define the prefixes for different file types
        prefixes = {
            'mp4': 'video_',
            'json': 'trajectory_',
            'txt': 'task_'
        }
        
        # Copy the files
        for file_type, prefix in prefixes.items():
            original_file_name = f"{prefix}{unique_identifier}.{file_type}"
            original_file_path = original_folder_path / original_file_name
            print(original_file_name)  # For demonstration purposes
            
            if not original_file_path.exists():
                raise FileNotFoundError(f"File does not exist: {original_file_path}")
            copy2(original_file_path, new_folder_path)
        print()  # For demonstration purposes


def copy_videos_of_category(excel_path, original_base_dir, new_base_dir, category):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_path)
    
    # Filter videos based on the specified category having a non-NaN value
    df_filtered = df.dropna(subset=[category])
    
    # Ensure the target directory exists
    os.makedirs(new_base_dir, exist_ok=True)
    
    # Iterate through the filtered DataFrame to copy each video
    for _, row in df_filtered.iterrows():
        video_name = row["Video Name"]
        if pd.notna(video_name):
            unique_identifier = video_name.replace('video_', '').rsplit('.', 1)[0]
            folder_index = unique_identifier.split('_', 1)[0]
            original_folder_path = pathlib.Path(original_base_dir) / folder_index

            prefixes = {
                'mp4': 'video_',
                'txt': 'task_'
            }
            
            for file_type, prefix in prefixes.items():
                original_file_name = f"{prefix}{unique_identifier}.{file_type}"
                original_file_path = original_folder_path / original_file_name
                print(original_file_name)
                
                if not original_file_path.exists():
                    raise FileNotFoundError(f"File does not exist: {original_file_path}")
                copy2(original_file_path, pathlib.Path(new_base_dir) / original_file_name)
            print() 



# ./runner.sh main/process_2d.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--app_output', default='main', type=str)
    app_group.add_argument('--result_name', default='/hdd2/chenyang/spatialpin/results', type=str)
    app_group.add_argument('--data_dir', default='/hdd2/chenyang/spatialpin/test_data', type=str)

    hparams = parser.parse_args()
    output_dir = hparams.result_name
    
    device_ = 'cuda:0'
    torch.cuda.set_device(device_)
    _CAMERA = camera.NOCS_Real()

    # Communicating to ChatGPT-4 API
    temperature_dict = {
      "obj_finding": 0.2,
      "task_proposal": 0.7,
      "spatial_understanding": 0.2,
      "motion_planning": 0.2,
      "code_generation": 0.2
    }
    # GPT-4 1106-preview is GPT-4 Turbo (https://openai.com/pricing)
    model_dict = {
      "obj_finding": "gpt-4-vision-preview",
      "task_proposal": "gpt-4-vision-preview",
      "spatial_understanding": "gpt-4-vision-preview",
      "motion_planning": "gpt-4-vision-preview",
      "code_generation": "gpt-4-vision-preview"
    }

    lang_sam_output_dir = pathlib.Path(output_dir) / "lang_sam"

    depth_estimation(hparams.data_dir, output_dir, device_)

    obj_finding_lang_sam(hparams.data_dir, str(lang_sam_output_dir), temperature_dict, model_dict, start_over=True)
    create_inpaint_masks(hparams.data_dir, str(lang_sam_output_dir))