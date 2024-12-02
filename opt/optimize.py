import os
import cv2
import imageio
import argparse
import pathlib
import copy
import time

import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.models.auto_encoder import PointCloudAE

from utils.nocs_utils import load_img_NOCS, create_input_norm, get_masks_out, get_aligned_masks_segout, get_masked_textured_pointclouds
from sdf_latent_codes.get_surface_pointcloud import get_sdfnet
from sdf_latent_codes.get_rgb import get_rgbnet
from utils.transform_utils import get_abs_pose_vector_from_matrix, get_abs_pose_from_vector, transform_pcd_to_canonical, transform_coordinates_3d, calculate_2d_projections, project, get_pc_absposes, transform_pcd_to_canonical
from utils.viz_utils import draw_colored_shape, draw_colored_mesh_mcubes, depth2inv, viz_inv_depth, save_projected_points, draw_bboxes_mpl_glow
from opt.optimization_all import Optimizer

from sdf_latent_codes.get_surface_pointcloud import get_surface_pointclouds_octgrid_viz, get_surface_pointclouds
from sdf_latent_codes.get_rgb import get_rgbnet, get_rgb_from_rgbnet

from utils.motion_utils import *
from utils.inpainting_utils import *

# from gpt_4.prompts.prompt_obj_idx_matching import match_object_index
# from gpt_4.prompts.prompt_spatial_understanding import understand_spatial_context
# from gpt_4.prompts.prompt_task_proposal import propose_task
# from gpt_4.prompts.prompt_motion_planning import plan_motion
# from gpt_4.prompts.prompt_code_generation import generate_code
# from gpt_4.prompts.utils import load_response, extract_code


def inference(
    hparams,
    data_dir, 
    output_path,
    min_confidence=0.1,
    use_gpu=True,
):
  model = PanopticModel(hparams, 0, None, None)
  model.eval()

  if use_gpu:
    model.cuda('cuda:3')
  data_path = open(os.path.join(data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
  sdf_pretrained_dir = os.path.join(data_dir, 'sdf_rgb_pretrained')
  rgb_model_dir = os.path.join(data_dir, 'sdf_rgb_pretrained', 'rgb_net_weights')
  _CAMERA = camera.NOCS_Real()
  min_confidence = 0.50
  sdf_pretrained_dir = os.path.join(data_dir, 'sdf_rgb_pretrained')
  rgb_model_dir = os.path.join(data_dir, 'sdf_rgb_pretrained', 'rgb_net_weights')
  
  # This line should be deleted
  # data_path = data_path[1]
  for i, img_path in enumerate(data_path):
    # Experiment on individual image
    if i != 2:
      continue
    
    img_full_path = os.path.join(data_dir, 'Real', img_path)
    # print(img_full_path)
    # print(img_path)
    color_path = img_full_path + '_color.png' 
    # print(color_path)

    if not os.path.exists(color_path):
      continue
    depth_full_path = img_full_path + '_depth.png'
    img_vis = cv2.imread(color_path)
    binding_box_vis = cv2.imread(str(pathlib.Path(output_path).parent / "inference" / f"box3d_{i}.png"))
    left_linear, depth, actual_depth = load_img_NOCS(color_path, depth_full_path)
    input = create_input_norm(left_linear, depth)
    input = input[None, :, :, :]
    if use_gpu:
      input = input.to(torch.device('cuda:3'))
    with torch.no_grad():
      seg_output, _, _ , pose_output = model.forward(input)
      shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, img_output, scores_out, output_indices = pose_output.compute_shape_pose_and_appearance(min_confidence,is_target = False)
      #shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices = nms(
      #  shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, scores_out, output_indices, _CAMERA
      #  )
    
    # get masks and masked pointclouds of each object in the image
    depth_ = np.array(depth, dtype=np.float32)*255.0
    seg_output.convert_to_numpy_from_torch()
    masks_out = get_masks_out(seg_output, depth_)
    masks_out = get_aligned_masks_segout(masks_out, output_indices, depth_)
    masked_pointclouds, areas, masked_rgb = get_masked_textured_pointclouds(masks_out, depth_, left_linear[:,:,::-1], camera = _CAMERA)

    optimization_out = {}
    latent_opt = []
    RT_opt = []
    scale_opt = []
    
    # abs_pose_opt
    abs_pose_opt = []
    # rgbnet_opt
    rgbnet_opt = []

    do_optim = True    
    latent_opt = []
    RT_opt = []
    scale_opt = []
    appearance_opt = []
    colored_opt_pcds = []
    colored_opt_meshes = []
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    psi, theta, phi, t = (0, 0, 0, 0)
    shape_latent_noise = np.random.normal(loc=0, scale=0.02, size=64)
    add_noise = False
    viz_type = None
    # psi, theta, phi, t = 0, 0, 4, 0.05
    if do_optim: 
      for k in range(len(shape_emb_outputs)):
          print("Starting optimization, object:", k, "\n", "----------------------------", "\n")
          if viz_type is not None:
            optim_foldername = str(output_path) + '/optim_images_'+str(k)
            if not os.path.exists(optim_foldername):
                os.makedirs(optim_foldername)
          else:
            optim_foldername = None
          # optimization starts here:
          abs_pose = abs_pose_outputs[k]
          mask_area = areas[k]
          RT, s = get_abs_pose_vector_from_matrix(abs_pose.camera_T_object, abs_pose.scale_matrix, add_noise = False)
          
          if masked_pointclouds[k] is not None:
            shape_emb = shape_emb_outputs[k]
            appearance_emb = appearance_emb_outputs[k]
            decoder = get_sdfnet(sdf_latent_code_dir = sdf_pretrained_dir)
            rgbnet = get_rgbnet(rgb_model_dir)
            params = {}
            weights = {}

            if add_noise:
              shape_emb += shape_latent_noise
            
            # Set latent vectors to optimize
            params['latent'] = shape_emb
            params['RT'] = RT
            params['scale'] = np.array(s)
            params['appearance'] = appearance_emb
            weights['3d'] = 1

            optimizer = Optimizer(params, rgbnet, device, weights, mask_area)
            # Optimize the initial pose estimate
            iters_optim = 200
            optimizer.optimize_oct_grid(
                iters_optim,
                masked_pointclouds[k],
                masked_rgb[k],
                decoder,
                rgbnet, 
                optim_foldername, 
                viz_type='3d'
            )
      
            # save latent vectors after optimization
            latent_opt.append(params['latent'].detach().cpu().numpy())
            RT_opt.append(params['RT'].detach().cpu().numpy())
            scale_opt.append(params['scale'].detach().cpu().numpy())
            appearance_opt.append(params['appearance'].detach().cpu().numpy())
            abs_pose = get_abs_pose_from_vector(params['RT'].detach().cpu().numpy(), params['scale'].detach().cpu().numpy())
            abs_pose_opt.append(abs_pose)
            rgbnet_opt.append(rgbnet)
            # obj_colored = draw_colored_shape(params['latent'].detach().cpu().numpy(), abs_pose, params['appearance'].detach().cpu().numpy(), rgbnet, sdf_pretrained_dir, is_oct_grid=True)
            # obj_colored = draw_colored_shape(latent_opt[k], abs_pose_opt[k], appearance_opt[k], rgbnet_opt[k], sdf_pretrained_dir, is_oct_grid=True)
            # obj_colored = draw_colored_mesh_mcubes(latent_opt[k], abs_pose_opt[k], appearance_opt[k], rgbnet_opt[k], output_path, "", i, k)
            # colored_opt_pcds.append(obj_colored)
          else:
            latent_opt.append(shape_emb_outputs[k])
            RT_opt.append(RT)
            scale_opt.append(np.array(s))
            appearance_opt.append(appearance_emb_outputs[k])
            abs_pose_opt.append(abs_pose)
            print("Done with optimization, object:", k, "\n", "----------------------------", "\n")
    
    # o3d.visualization.draw_geometries(colored_opt_pcds)
    # colored_opt_pcds is the list of point clouds of objects within the image
    # o3d.io.write_point_cloud can only save one object

    # for k, pcd in enumerate(colored_opt_pcds):
    #   # Add saving 3D point clouds file here
    #   filename = 'colored_opt_pcds_' + str(i) + "_pose_" + str(k)+'.pcd'
    #   o3d.io.write_point_cloud(str(output_path) + '/' + filename, pcd)

    #   # Add saving projection of 3D points clouds to 2D RGB image here
    #   filename = 'projected_' + str(i) + "_pose_" + str(k)+'.png'
    #   img_projected = project_colored_pc_to_img(pcd, _CAMERA.K_matrix[:3,:3], img_vis, True)
    #   cv2.imwrite(str(output_path) + '/' + filename, img_projected)

    transformed_axes_opt = []
    for k, pcd in enumerate(abs_pose_opt):
      # transformed_axes
      transformed_axes_opt.append(calculate_transformed_axes(abs_pose_opt[k]))

      # mask for each object for image inpainting
      # filename = 'mask_' + str(i) + "_pose_" + str(k)+'.png'
      # save_projected_mask(pcd, _CAMERA.K_matrix[:3,:3], img_vis, filename, erosion_pixels=20)
    

    # Observe object 6D poses and local axes
    # print(_CAMERA.K_matrix[:3,:3])
    # print(abs_pose_opt[4])
    # print(type(abs_pose_opt[0]))
    # for k, _ in enumerate(abs_pose_opt):
    #   print("Object", k)
    #   print(abs_pose_opt[k].camera_T_object)
    #   print(abs_pose_opt[k].scale_matrix)
    #   print(transformed_axes_opt[k])
    #   print(" ")

    # Keypoints generation (aborted)
    # keypoints_list = generate_keypoints(colored_opt_pcds)
    # vis_keypoints_on_img(img_vis, i, output_path, abs_pose_opt, keypoints_list, _CAMERA.K_matrix[:3,:3])


    # Information for ChatGPT-4
    results_2d = print_local_axis_center(abs_pose_opt, transformed_axes_opt, _CAMERA.K_matrix[:3,:3], print_only_2d=True)
    results_3d = print_local_axis_center(abs_pose_opt, transformed_axes_opt, _CAMERA.K_matrix[:3,:3], print_only_2d=False)
    info_obj_size = print_obj_size(abs_pose_opt, colored_opt_pcds)
    info_spatial_relation = print_spatial_relation(abs_pose_opt, print_only_closest=False)

    # Communicating to ChatGPT-4 API
    temperature_dict = {
      "obj_idx_matching": 0.2,
      "spatial_understanding": 0.2,
      "task_proposal": 0.7,
      "motion_planning": 0.2,
      "code_generation": 0.2
    }
    # GPT-4 1106-preview is GPT-4 Turbo (https://openai.com/pricing)
    model_dict = {
      "obj_idx_matching": "gpt-4-vision-preview",
      "spatial_understanding": "gpt-4-vision-preview",
      "task_proposal": "gpt-4-vision-preview",
      "motion_planning": "gpt-4-vision-preview",
      "code_generation": "gpt-4-vision-preview"
    }
    
    conversation_hist = []
    obj_idx_matching_path = pathlib.Path(output_path).parent / "gpt4_response" / "prompts/obj_idx_matching" / str(i)
    spatial_understanding_path = pathlib.Path(output_path).parent / "gpt4_response" / "prompts/spatial_understanding" / str(i)
    task_proposal_path = pathlib.Path(output_path).parent / "gpt4_response" / "prompts/task_proposal" / str(i)
    motion_planning_path = pathlib.Path(output_path).parent / "gpt4_response" / "prompts/motion_planning" / str(i)
    code_generation_path = pathlib.Path(output_path).parent / "gpt4_response" / "prompts/code_generation" / str(i)

    start_over = False
    if start_over:
      user_0, res_0 = match_object_index(i, img_vis, binding_box_vis, results_2d, str(output_path), existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
      conversation_hist.append((user_0, res_0))
      user_1, res_1 = understand_spatial_context(i, img_vis, binding_box_vis, results_3d + info_obj_size + info_spatial_relation, str(output_path), existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
      conversation_hist.append((user_1, res_1))

      for obj_idx in range(len(abs_pose_opt)):
        user_2, res_2 = propose_task(i, img_vis, binding_box_vis, obj_idx, str(output_path), existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        conversation_hist.append((user_2, res_2))
        time.sleep(30)
        user_3, res_3 = plan_motion(i, img_vis, binding_box_vis, None, str(output_path), existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        conversation_hist.append((user_3, res_3))
        time.sleep(30)
        generate_code(i, img_vis, binding_box_vis, None, str(output_path), existing_response=None, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        conversation_hist = conversation_hist[:-2]

    else:
      user_0, res_0 = match_object_index(i, img_vis, binding_box_vis, results_2d, str(output_path), existing_response=load_response("obj_idx_matching", obj_idx_matching_path), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
      conversation_hist.append((user_0, res_0))
      user_1, res_1 = understand_spatial_context(i, img_vis, binding_box_vis, results_3d + info_obj_size + info_spatial_relation, str(output_path), existing_response=load_response("spatial_understanding", spatial_understanding_path), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
      conversation_hist.append((user_1, res_1))

      task_responses = load_response("task_proposal", task_proposal_path, get_latest=False)
      for k, task_response in enumerate(task_responses): 
        user_2, res_2 = propose_task(i, img_vis, binding_box_vis, k % len(abs_pose_opt), str(output_path), existing_response=task_response, temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        conversation_hist.append((user_2, res_2))
        user_3, res_3 = plan_motion(i, img_vis, binding_box_vis, None, str(output_path), existing_response=load_response("motion_planning", motion_planning_path, file_idx=k, get_latest=False), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        conversation_hist.append((user_3, res_3))
        generate_code(i, img_vis, binding_box_vis, None, str(output_path), existing_response=load_response("code_generation", code_generation_path, file_idx=k, get_latest=False), temperature_dict=temperature_dict, model_dict=model_dict, conversation_hist=conversation_hist)
        conversation_hist = conversation_hist[:-2]



    def generate_video(obj_idx, motion_list, img_vis, latent_opt, abs_pose_opt, appearance_opt, rgbnet_opt, sdf_pretrained_dir, camera_matrix, video_path, video_name):
      motion_steps = generate_seq_motion_physical_rrt(obj_idx, motion_list, abs_pose_opt, transformed_axes_opt, colored_opt_pcds, camera_matrix)
      images = []
      pcds_to_proj = []
      pcds_idx_to_proj = extract_pcds_idx_to_proj(obj_idx, motion_list)
      
      # inpainted_img = load_inpainted_img(i, obj_idx)
      # inpainted_img = inpaint_img_from_impainted(img_vis, i, list(range(len(abs_pose_opt))), colored_opt_pcds, camera_matrix)
      inpainted_img = inpaint_img_from_impainted(img_vis, i, pcds_idx_to_proj, colored_opt_pcds, camera_matrix)

      for idx in range(len(abs_pose_opt)):
        if idx in pcds_idx_to_proj and idx != obj_idx:
          pcds_to_proj.append(draw_colored_shape(latent_opt[idx], abs_pose_opt[idx], appearance_opt[idx], rgbnet_opt[idx], sdf_pretrained_dir, is_oct_grid=True))
          # pcds_to_proj.append(draw_colored_mesh_mcubes(latent_opt[idx], abs_pose_opt[idx], appearance_opt[idx], rgbnet_opt[idx], output_path, "", i, idx))
        
      for k, abs_pose in enumerate(motion_steps):
        pcd = draw_colored_shape(latent_opt[obj_idx], abs_pose, appearance_opt[obj_idx], rgbnet_opt[obj_idx], sdf_pretrained_dir, is_oct_grid=True)
        # pcd = draw_colored_mesh_mcubes(latent_opt[obj_idx], abs_pose, appearance_opt[obj_idx], rgbnet_opt[obj_idx], output_path, "", i, obj_idx)
        img_motion = project_colored_pc_to_img_depth(pcds_to_proj + [pcd], camera_matrix, inpainted_img, True)
        images.append(img_motion)


      # Convert the color format from BGR (used by OpenCV) to RGB
      for k, img_motion in enumerate(images):
        # Save generated images separately
        images[k] = cv2.cvtColor(img_motion, cv2.COLOR_BGR2RGB)
      
      # Save as a GIF
      imageio.mimsave(str(video_path / video_name), images, duration=0.1)  # duration controls frame delay



    video_path = pathlib.Path(output_path).parent / "video"
    generated_codes, video_filenames = extract_code("code_generation", code_generation_path, video_path, i)

    for k, generated_code in enumerate(generated_codes):
      obj_idx, motion_list = generated_code[0], generated_code[1]
      video_name = video_filenames[k]
      generate_video(obj_idx, motion_list, img_vis, latent_opt, abs_pose_opt, appearance_opt, rgbnet_opt, sdf_pretrained_dir, _CAMERA.K_matrix[:3,:3], video_path, video_name)

    # obj_idx = 3
    # motion_list = [("translate_tar_obj", [-5, -10, 5], 1), ("rotate_wref", "fixed_towards", "pitch", 1), ("rotate_wref", "fixed_towards", "yaw", 1), ("rotate_wref", "fixed_towards", "roll", 1),("rotate_wref", "fixed_back", "roll", 1), ("rotate_wref", "fixed_back", "yaw", 1), ("rotate_wref", "fixed_back", "pitch", 1), ("translate_tar_obj", [0, 0, 0], 3)]
    # motion_list = [("translate_tar_obj", [-5, 5, 5], 1), ("rotate_wref", "fixed_towards", "pitch", 1), ("rotate_wref", "fixed_back", "pitch", 1), ("translate_tar_obj", [0, 0, 0], 3)]
    # motion_list = [("translate_tar_obj", [0, 4.5, 0], 1)]
    # motion_list = [("translate_direc_axis", 10, 1, 4)]
    # generate_video(obj_idx, motion_list, img_vis, latent_opt, abs_pose_opt, appearance_opt, rgbnet_opt, sdf_pretrained_dir, _CAMERA.K_matrix[:3,:3], video_path, f"gen_motion_{i}.gif")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='optimize', type=str)
  app_group.add_argument('--result_name', default='ShAPO_Real', type=str)
  app_group.add_argument('--data_dir', default='nocs_data', type=str)

  hparams = parser.parse_args()
  # print(hparams)
  result_name = hparams.result_name
  path = 'results/'+result_name
  output_path = pathlib.Path(path) / hparams.app_output
  output_path.mkdir(parents=True, exist_ok=True)
  inference(hparams, hparams.data_dir, output_path)
