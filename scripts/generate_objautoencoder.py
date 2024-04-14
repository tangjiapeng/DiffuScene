"""Script used to train a ATISS."""
import argparse
from fileinput import filename
import logging
import os
import sys
from wsgiref import validate

import numpy as np

import torch
from torch.utils.data import DataLoader

from training_utils import id_generator, save_experiment_params, load_config

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from scene_synthesis.stats_logger import StatsLogger, WandB

from utils import yield_forever, load_checkpoints, save_checkpoints
from scene_synthesis.networks.foldingnet_autoencoder import AutoEncoder, KLAutoEncoder, train_on_batch, validate_on_batch
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureNormPCDataset
from scene_synthesis.datasets.utils_io import export_pointcloud, load_pointcloud

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )
    ###
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    ##
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )

    # Parse the config file
    config = load_config(args.config_file)

    scenes_train_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config["data"], config["training"].get("splits", ["train", "val"]), config["data"]["without_lamps"])
    )
    print("Loading train dataset with {} rooms".format(len(scenes_train_dataset)))

    scenes_validation_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config["data"], config["validation"].get("splits", ["test"]), config["data"]["without_lamps"])
    )
    print("Loading validation dataset with {} rooms".format(len(scenes_validation_dataset)))

    # Collect the set of objects in the scenes
    train_objects = {}
    for scene in scenes_train_dataset:
        for obj in scene.bboxes:
            train_objects[obj.model_jid] = obj
    train_objects = [vi for vi in train_objects.values()]
    train_dataset = ThreedFutureNormPCDataset(train_objects)

    validation_objects = {}
    for scene in scenes_validation_dataset:
        for obj in scene.bboxes:
            validation_objects[obj.model_jid] = obj
    validation_objects = [vi for vi in validation_objects.values()]
    validation_dataset = ThreedFutureNormPCDataset(validation_objects)


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} train objects".format(
        len(train_dataset))
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation objects".format(
        len(validation_dataset))
    )

    # Build the network architecture to be used for training
    ### instead of using build_network, we directly build from config
    network = KLAutoEncoder(latent_dim=config["network"].get("objfeat_dim", 64),  kl_weight=config["network"].get("kl_weight", 0.001))
    if args.weight_file is not None:
        print("Loading weight file from {}".format(args.weight_file))
        network.load_state_dict(
            torch.load(args.weight_file, map_location=device)
        )
    network.to(device)
    ####
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}:  {n_trainable_params} / {n_all_params}")

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], filter(lambda p: p.requires_grad, network.parameters()) ) 

    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Load the learning rate scheduler 
    lr_scheduler = schedule_factory(config["training"])

    generation_directory = os.path.join(
        args.output_directory,
        experiment_tag,
        "generation"
    )
    if not os.path.exists(generation_directory):
        os.makedirs(generation_directory)

    lat_list = []
    with torch.no_grad():
        print("====> Validation Epoch ====>")
        network.eval()
        for b, sample in enumerate(val_loader):
            # Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            kl, lat, rec = network(sample["points"])
            idx = sample["idx"]
            lat_list.append(lat)

            for i in range(lat.shape[0]):
                lat_i = lat[i].cpu().numpy()
                pc_i = sample["points"][i].cpu().numpy()
                rec_i = rec[i].cpu().numpy()
                idx_i = idx[i].item()

                # save obj autoencoder results for vis check
                model_jid = validation_dataset.get_model_jid(idx_i)["model_jid"]
                filename_input = "{}/{}.ply".format(generation_directory, model_jid)
                filename_rec  =  "{}/{}_rec.ply".format(generation_directory, model_jid)
                export_pointcloud(pc_i, filename_input)
                export_pointcloud(rec_i, filename_rec)


                latent_dim=config["network"].get("objfeat_dim", 64)
                #save objfeat i.e. latent
                obj = validation_dataset.objects[idx_i]
                assert model_jid == obj.model_jid
                raw_model_path = obj.raw_model_path
                filename_lats = raw_model_path[:-4] + "_norm_pc_lat{:d}.npz".format(latent_dim)
                np.savez(filename_lats, latent=lat_i)

            print('iter {}'.format(b), lat.shape, lat.min(), lat.max(), rec.shape)
            
        lat_all = torch.cat(lat_list, dim=0)
        print('before: std {}, min {}, max {}'.format(lat_all.flatten().std(), lat_all.min(), lat_all.max()) )
        scale_factor = 1.0 / lat_all.flatten().std()
        print('scale factor:', scale_factor)
        lat_scaled = lat_all * scale_factor
        print('after: std {}, min {}, max {}'.format(lat_scaled.flatten().std(), lat_scaled.min(), lat_scaled.max()) )
        print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])

