"""Script used to train a Shape Autoencoder."""
import argparse
import logging
import os
import sys
from wsgiref import validate

import numpy as np

import torch
from torch.utils.data import DataLoader

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from scene_synthesis.stats_logger import StatsLogger, WandB
from training_utils import id_generator, save_experiment_params, load_config, yield_forever, load_checkpoints, save_checkpoints

from scene_synthesis.networks.foldingnet_autoencoder import AutoEncoder, KLAutoEncoder, train_on_batch, validate_on_batch
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureNormPCDataset


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )
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
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
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
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    scenes_train_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config["data"], config["training"].get("splits", ["train", "val"]), config["data"]["without_lamps"])
    )
    print("Loading train dataset with {} rooms".format(len(scenes_train_dataset)))


    # add dining rooms
    config2 = {
        "filter_fn":                 "threed_front_diningroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config["data"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config["data"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "../config/diningroom_threed_front_splits.csv"
    }
    scenes_train_dataset2 = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config2, config["training"].get("splits", ["train", "val"]), config["data"]["without_lamps"])
    )
    print("Loading train dataset 2 with {} rooms".format(len(scenes_train_dataset2)))

    ## add living rooms
    config3 = {
        "filter_fn":                 "threed_front_livingroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config["data"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config["data"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "../config/livingroom_threed_front_splits.csv"
    }
    scenes_train_dataset3 = ThreedFront.from_dataset_directory(
        dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["data"]["path_to_model_info"],
        path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config3, config["training"].get("splits", ["train", "val"]), config["data"]["without_lamps"])
    )
    print("Loading train dataset 3 with {} rooms".format(len(scenes_train_dataset3)))

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
    # diningroom
    for scene in scenes_train_dataset2:
        for obj in scene.bboxes:
            train_objects[obj.model_jid] = obj
    # livingroom
    for scene in scenes_train_dataset3:
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
    print("Loaded {} train objects of bedroom/livingroom/dining".format(
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

    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get(
                "project", "autoregressive_transformer"
            ),
            name=experiment_tag,
            watch=False,
            log_frequency=10
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        # adjust learning rate
        adjust_learning_rate(lr_scheduler, optimizer, i)

        network.train()
        #for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
        for b, sample in enumerate(train_loader):
            # Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            batch_loss = train_on_batch(network, optimizer, sample, config)
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)

        if (i % save_every) == 0:
            save_checkpoints(
                i,
                network,
                optimizer,
                experiment_directory,
            )
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    if not isinstance(v, list):
                        sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample, config)
                StatsLogger.instance().print_progress(-1, b+1, batch_loss)
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])
