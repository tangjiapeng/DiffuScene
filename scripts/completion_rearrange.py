"""Script to perform scene completion and rearrangement."""
import argparse
import logging
import os
from random import sample
import sys

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, get_textured_objects_in_scene, \
    poll_specific_class, make_network_input, render_to_folder, \
    render_scene_from_bbox_params

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects, get_textured_objects_based_on_objfeats

from simple_3dviz import Scene
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
from pyrr import Matrix44
from utils import render as render_top2down
from scene_synthesis.stats_logger import AverageAggregator
from utils import merge_meshes,  computer_intersection, computer_symmetry
import trimesh

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default="../demo/floor_plan_texture_images",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )
    ####
    parser.add_argument(
        "--render_top2down",
        action="store_true",
        help="Perform top2down orthographic rendering"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="if remove the floor plane"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="if remove the texture"
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="if save mesh"
    )
    parser.add_argument(
        "--mesh_format",
        type=str,
        default=".ply",
        help="mesh format "
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="if clip_denoised"
    )
    parser.add_argument(
        "--retrive_objfeats",
        action="store_true",
        help="if retrive most similar objectfeats"
    )
    ###
    parser.add_argument(
        "--num_partial",
        type=int,
        default=3,
        help="number of objects in partial scenes "
    )
    parser.add_argument(
        "--arrange_objects",
         action="store_true",
        help="if use retrived scale"
    )
    parser.add_argument(
        "--scene_texture",
        action="store_true",
        help="if remove the texture"
    )
    parser.add_argument(
        "--compute_intersec",
        action="store_true",
        help="if remove the texture"
    )


    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    ########## make it for evaluation
    if 'text' in config["data"]["encoding_type"]:
        if 'textfix' not in config["data"]["encoding_type"]:
            config["data"]["encoding_type"] = config["data"]["encoding_type"].replace('text', 'textfix')

    if "no_prm" not in config["data"]["encoding_type"]:
        print('NO PERM AUG in test')
        config["data"]["encoding_type"] = config["data"]["encoding_type"] + "_no_prm"
    print('encoding type :', config["data"]["encoding_type"])
    ####### 
    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))


    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    # Create the scene and the behaviour list for simple-3dviz top-down orthographic rendering, the arguments are same as preprocess_data.py
    if args.without_floor:
        scene_top2down = Scene(size=(256, 256), background=[1,1,1,1])
    else:
        scene_top2down = Scene(size=(256, 256), background=[0,0,0,1])
    scene_top2down.up_vector = (0,0,-1)
    scene_top2down.camera_target = (0, 0, 0)
    scene_top2down.camera_position = (0,4,0)
    scene_top2down.light = (0,4,0)
    scene_top2down.camera_matrix = Matrix44.orthogonal_projection(
        left=-3.1, right=3.1,
        bottom=3.1, top=-3.1,
        near=0.1, far=6
    )

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i
                
                
    if args.compute_intersec:
        num_objects_counter = []
        total_num_symmetry, total_num_pairs = 0, 0
        NUM_OBJ = AverageAggregator()
        NUM_PAIRS = AverageAggregator()
        BOX_IOU = AverageAggregator()
        BOX_INSEC = AverageAggregator()
        OVERLAP_RATIO = AverageAggregator()

        num_objects_counter_onlysize = []
        total_num_symmetry_onlysize, total_num_pairs_onlysize  = 0, 0
        NUM_OBJ_ONLYSIZE = AverageAggregator()
        NUM_PAIRS_ONLYSIZE = AverageAggregator()
        BOX_IOU_ONLYSIZE = AverageAggregator()
        BOX_INSEC_ONLYSIZE= AverageAggregator()
        OVERLAP_RATIO_ONLYSIZE = AverageAggregator()

    classes = np.array(dataset.class_labels)
    print('class labels:', classes, len(classes))
    for i in range(args.n_sequences):
        if args.arrange_objects:
            if i < len(dataset):
                scene_idx = given_scene_id or i
            else:
                scene_idx = given_scene_id or (i % len(dataset))
        else:
            scene_idx = given_scene_id or np.random.choice(len(dataset))
        current_scene = raw_dataset[scene_idx]
        samples = dataset[scene_idx]
        print("{} / {}: Using the {} floor plan of scene {}".format(
            i, args.n_sequences, scene_idx, current_scene.scene_id)
        )
        # Get a floor plan
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures, no_texture=args.no_texture
        )
        

        if args.arrange_objects:
            num_boxes, angle_dim = samples["angles"].shape
            print( 'number of boxes are {:d}'.format(num_boxes) )
            if args.scene_id:
                #np.random.seed(scene_idx); 
                noise_trans = 0.4 * np.random.randn(num_boxes, 3).astype(np.float32) # * np.array([[1, 0, 0]]).repeat(num_boxes, axis=0).astype(np.float32)
                samples["translations"][:, 0:1] = noise_trans[:, 0:1]
                samples["translations"][:, 2:3] = noise_trans[:, 2:3]
            else:
                # add random noisy translation for visualization
                np.random.seed(scene_idx); noise_trans = np.random.randn(num_boxes, 3).astype(np.float32)
                samples["translations"] =  samples["translations"] + noise_trans
            # add random noisy size for visualization
            np.random.seed(scene_idx + len(dataset)); noise_angle = np.random.randn(num_boxes, angle_dim).astype(np.float32)
            samples["angles"] =  samples["angles"] + noise_angle
            print('Adding random noise to trans and angles for visualizaitons')

            if config["network"].get("objectness_dim", 0) >0:
                num_objects = int((samples["objectness"]>0).sum())
            else:
                num_objects = num_boxes - int((samples["class_labels"][:, -1]>0).sum())
            print('number of nonempty objects {}'.format(num_objects))
            num_partial = num_boxes
            render_foldername = "noisy"
        else:
            num_partial = args.num_partial
            print('number of objects in partial scenes {}'.format(num_partial))
            render_foldername = "partial"

        bbox_params = {
            "class_labels": torch.from_numpy(samples["class_labels"])[None, :num_partial, ...],
            "translations": torch.from_numpy(samples["translations"])[None, :num_partial, ...],
            "sizes": torch.from_numpy(samples["sizes"])[None, :num_partial, ...],
            "angles": torch.from_numpy(samples["angles"])[None, :num_partial, ...],
        }
        input_boxes = torch.cat([
                bbox_params["translations"],
                bbox_params["sizes"],
                bbox_params["angles"],
                bbox_params["class_labels"],
            ], dim=-1
        )
        if config["network"].get("objectness_dim", 0) >0:
            bbox_params["objectness"] = torch.from_numpy(samples["objectness"])[None, :num_partial, ...]
            input_boxes = torch.cat([ input_boxes, bbox_params["objectness"] ], dim=-1)

        if config["network"].get("objfeat_dim", 0) >0:
            if config["network"]["objfeat_dim"] == 32:
                bbox_params["objfeats"] = torch.from_numpy(samples["objfeats_32"])[None, :num_partial, ...]
            else:
                bbox_params["objfeats"] = torch.from_numpy(samples["objfeats"])[None, :num_partial, ...]
            input_boxes = torch.cat([ input_boxes, bbox_params["objfeats"] ], dim=-1)
        
        if args.arrange_objects:
            # delete empty object
            bbox_params = network.delete_empty_boxes(bbox_params, device=device)

        # Render the failed scene
        render_to_folder(
            args,
            render_foldername,
            "{}_{}_{:03}".format(current_scene.scene_id, scene_idx, i),
            dataset,
            objects_dataset,
            tr_floor,
            floor_plan,
            scene,
            scene_top2down,
            bbox_params,
            add_start_end=False,
            diffusion=True,
        )
         
        if args.arrange_objects:
            bbox_params = network.arrange_scene(
                    room_mask=room_mask.to(device),
                    num_points=config["network"]["sample_num_points"],
                    point_dim=config["network"]["point_dim"],
                    input_boxes=input_boxes.to(device),
                    device=device,
                    clip_denoised=args.clip_denoised,
                    batch_seeds=torch.arange(i, i+1),
            )
        else:
            bbox_params = network.complete_scene(
                    room_mask=room_mask.to(device),
                    num_points=config["network"]["sample_num_points"],
                    point_dim=config["network"]["point_dim"],
                    partial_boxes=input_boxes.to(device),
                    device=device,
                    clip_denoised=args.clip_denoised,
                    batch_seeds=torch.arange(i, i+1),
            )
    
        
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()
        print('Generated bbox:', bbox_params_t.shape)


        if args.retrive_objfeats:
            objfeats = boxes["objfeats"].cpu().numpy()
            print('shape retrieval based on obj latent feats')

            renderables, trimesh_meshes, model_jids = get_textured_objects_based_on_objfeats(
                bbox_params_t, objects_dataset, classes, diffusion=True, no_texture=args.no_texture, query_objfeats=objfeats, 
            )

            renderables_onlysize, trimesh_meshes_onlysize, model_jids_onlysize = get_textured_objects(
                    bbox_params_t, objects_dataset, classes, diffusion=True, no_texture=args.no_texture
                )
        else:
            renderables, trimesh_meshes, model_jids = get_textured_objects(
                bbox_params_t, objects_dataset, classes, diffusion=True, no_texture=args.no_texture
            )

        if args.compute_intersec:
            num_objects, num_pairs, avg_iou, avg_insec, overlap_ratio = computer_intersection(trimesh_meshes)
            num_objects_counter.append(num_objects)
            NUM_OBJ.value = num_objects
            NUM_PAIRS.value = num_pairs
            BOX_IOU.value = avg_iou
            BOX_INSEC.value = avg_insec
            OVERLAP_RATIO.value = overlap_ratio
            num_symmetries = computer_symmetry(trimesh_meshes, boxes["class_labels"][0, :, :].cpu().numpy(), model_jids)
            total_num_symmetry += num_symmetries
            total_num_pairs += num_pairs
            string = "num scenes: {:d} - num objects avg: {:f} - std: {:f} - num pairs: {:f} - box iou: {:f} - box intersec: {:f} - overlap ratio: {:f} - total num symmetries: {:d} - total num pairs: {:d} retrive objfeats  "\
                .format(i+1, NUM_OBJ.value, np.array(num_objects_counter).std(), NUM_PAIRS.value, BOX_IOU.value, BOX_INSEC.value, OVERLAP_RATIO.value, total_num_symmetry, total_num_pairs)
            print( string )
            with open( os.path.join(args.output_directory, "iou_states.txt"), 'a') as f:
                f.write(string+"\n")
            f.close()

            if args.retrive_objfeats:
                num_objects, num_pairs, avg_iou, avg_insec, overlap_ratio = computer_intersection(trimesh_meshes_onlysize)
                num_objects_counter_onlysize.append(num_objects)
                NUM_OBJ_ONLYSIZE.value = num_objects
                NUM_PAIRS_ONLYSIZE.value = num_pairs
                BOX_IOU_ONLYSIZE.value = avg_iou
                BOX_INSEC_ONLYSIZE.value = avg_insec
                OVERLAP_RATIO_ONLYSIZE.value = overlap_ratio

                num_symmetries = computer_symmetry(trimesh_meshes_onlysize, boxes["class_labels"][0, :, :].cpu().numpy(), model_jids_onlysize)
                total_num_symmetry_onlysize += num_symmetries
                total_num_pairs_onlysize += num_pairs
                string = "num scenes: {:d} - num objects avg: {:f} - std: {:f} - num pairs: {:f} - box iou: {:f} - box intersec: {:f} - overlap ratio: {:f} - total num symmetries: {:d} - total num pairs: {:d} retrive only size"\
                .format(i+1, NUM_OBJ_ONLYSIZE.value, np.array(num_objects_counter_onlysize).std(), NUM_PAIRS_ONLYSIZE.value, BOX_IOU_ONLYSIZE.value, BOX_INSEC_ONLYSIZE.value, OVERLAP_RATIO_ONLYSIZE.value, total_num_symmetry_onlysize, total_num_pairs_onlysize)
                print( string )
                with open( os.path.join(args.output_directory, "iou_states.txt"), 'a') as f:
                    f.write(string+"\n")
                f.close()
        
           
        # Specify the path of the rendered image
        path_to_image = "{}/{}_{}_{:03d}.png".format(
            args.output_directory,
            current_scene.scene_id,
            scene_idx,
            i
        )
        # Specify the path to the save the generated scene
        path_to_objs = os.path.join(
            args.output_directory,
            "complete",
        )
        render_scene_from_bbox_params(
            args,
            bbox_params,
            dataset,
            objects_dataset,
            classes,
            floor_plan, 
            tr_floor,
            scene,
            scene_top2down,
            path_to_image,
            path_to_objs,
            "{}_{}_{:03d}".format(current_scene.scene_id,  scene_idx, i),
            network_objae=None,
            device=device,
            diffusion=True,
        )

        # generate target
        if config["validation"]["gen_gt"]:
            samples = dataset[scene_idx]

            print("{} / {} gt: Using the {} floor plan of scene {}".format(
                i, args.n_sequences, scene_idx, current_scene.scene_id)
            )
            
            # generate predicted scene objects
            print("Ground-truth shape:", samples["class_labels"].shape)
            bbox_params = {
            "class_labels": torch.from_numpy(samples["class_labels"])[None, :, ...],
            "translations": torch.from_numpy(samples["translations"])[None, :, ...],
            "sizes": torch.from_numpy(samples["sizes"])[None, :, ...],
            "angles": torch.from_numpy(samples["angles"])[None, :, ...],
            }
            if config["network"].get("objectness_dim", 0) >0:
                bbox_params["objectness"] = torch.from_numpy(samples["objectness"])[None, :, ...]
            if config["network"].get("objfeat_dim", 0) >0:
                if config["netowrk"]["objfeat_dim"] == 32:
                    bbox_params["objfeats"] = torch.from_numpy(samples["objfeats_32"])[None, :, ...]
                else:
                    bbox_params["objfeats"] = torch.from_numpy(samples["objfeats"])[None, :, ...]

            bbox_params = network.delete_empty_boxes(bbox_params, device=device)

            # Render the ground-truth scene
            render_to_folder(
                args,
                "groundtruth",
               "{}_{}_{:03d}".format(current_scene.scene_id,  scene_idx, i),
                dataset,
                objects_dataset,
                tr_floor,
                floor_plan,
                scene,
                scene_top2down,
                bbox_params,
                add_start_end=False,
                diffusion=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
