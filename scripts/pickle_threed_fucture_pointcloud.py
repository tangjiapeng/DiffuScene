"""Script used for pickling the 3D Future dataset in order to be subsequently
used by our scripts.
"""
import argparse
from concurrent.futures import process
from fileinput import filename
import os
import sys

import pickle

from scene_synthesis.datasets import filter_function
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_front_dataset import dataset_encoding_factory
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
import trimesh
import numpy as np

from scene_synthesis.datasets.utils_io import export_pointcloud

def main(argv):
    parser = argparse.ArgumentParser(
        description="Pickle the 3D Future dataset"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )
    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
    )
    parser.add_argument(
        "--dataset_filtering",
        default="threed_front_bedroom",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library"
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )
    ##
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=0.0,
        help="The bbox padding, default 0.0 as occnet"
    )
    parser.add_argument(
        '--pointcloud_size', 
        type=int, default=30000,
        help='Size of point cloud.')
    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    with open(args.path_to_invalid_scene_ids, "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    with open(args.path_to_invalid_bbox_jids, "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        "annotation_file":           args.annotation_file
    }

    for split in ["train", "val", "test"]:

        # Initially, we only consider the train split to compute the dataset
        # statistics, e.g the translations, sizes and angles bounds
        scenes_dataset = ThreedFront.from_dataset_directory(
            dataset_directory=args.path_to_3d_front_dataset_directory,
            path_to_model_info=args.path_to_model_info,
            path_to_models=args.path_to_3d_future_dataset_directory,
            filter_fn=filter_function(config, [split], args.without_lamps)
        )
        print("Loading dataset with {} rooms".format(len(scenes_dataset)))

        # Collect the set of objects in the scenes
        objects = {}
        for scene in scenes_dataset:
            for obj in scene.bboxes:
                objects[obj.model_jid] = obj
        objects = [vi for vi in objects.values()]

        objects_dataset = ThreedFutureDataset(objects)
        room_type = args.dataset_filtering.split("_")[-1]
        output_directory = "{}/threed_future_pointcloud_{}".format(
            args.output_directory,
            room_type
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        model_names = []
        #read the ThreedFutureDataset
        for idx in range(len(objects_dataset)):
            obj = objects_dataset[idx]
            model_uid = obj.model_uid
            model_jid = obj.model_jid
            raw_model_path = obj.raw_model_path
            model_names.append(model_jid)

            # read mesh
            mesh = trimesh.load(
                raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,
                skip_texture=True
            )
            bbox = mesh.bounding_box.bounds

            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

            # Transform input mesh
            mesh.apply_translation(-loc)
            mesh.apply_scale(1 / scale)


            # sample point clouds with normals
            points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
            normals = mesh.face_normals[face_idx]

            # Compress
            dtype = np.float16
            #dtype = np.float32
            points = points.astype(dtype)
            normals = normals.astype(dtype)

            # save_npz or ply
            #filename = "{}/{}.npz".format(output_directory, model_jid)
            filename = raw_model_path[:-4] + "_norm_pc.npz"
            print('Writing pointcloud: %s' % filename)
            np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)

            filename = "{}/{}.ply".format(output_directory, model_jid)
            export_pointcloud(points, filename)
            print('Writing pointcloud: %s' % filename)

        # write train/val/test split
        split_lst = "{}/threed_future_pointcloud_{}/{}.lst".format(
            args.output_directory,
            room_type,
            split
        )
        open(split_lst, 'w').writelines([ name +'\n' for name in model_names])
            

    

if __name__ == "__main__":
    main(sys.argv[1:])