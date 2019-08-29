#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.ERROR)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Declare some parameters
        batch_size = 32

        # Define some model-specific parameters
        elg_first_layer_stride = 1
        elg_num_modules = 3
        elg_num_feature_maps = 32

        # Define training data source
        from datasources import UnityEyes
        unityeyes = UnityEyes(
            session,
            batch_size=batch_size,
            data_format='NCHW',
            unityeyes_path='../datasets/UnityEyes/imgs',
            min_after_dequeue=1000,
            generate_heatmaps=True,
            shuffle=True,
            staging=True,
            eye_image_shape=(36, 60),
            heatmaps_scale=1.0 / elg_first_layer_stride,
        )
        unityeyes.set_augmentation_range('translation', 2.0, 10.0)
        unityeyes.set_augmentation_range('rotation', 1.0, 10.0)
        unityeyes.set_augmentation_range('intensity', 0.5, 20.0)
        unityeyes.set_augmentation_range('blur', 0.1, 1.0)
        unityeyes.set_augmentation_range('scale', 0.01, 0.1)
        unityeyes.set_augmentation_range('rescale', 1.0, 0.5)
        unityeyes.set_augmentation_range('num_line', 0.0, 2.0)
        unityeyes.set_augmentation_range('heatmap_sigma', 7.5, 2.5)

        # Define model
        from models import ELG
        model = ELG(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # Model configuration parameters
            # first_layer_stride describes how much the input image is downsampled before producing
            #                    feature maps for eventual heatmaps regression
            # num_modules defines the number of hourglass modules, and thus the number of times repeated
            #             coarse-to-fine refinement is done.
            # num_feature_maps describes how many feature maps are refined over the entire network.
            first_layer_stride=elg_first_layer_stride,
            num_feature_maps=elg_num_feature_maps,
            num_modules=elg_num_modules,

            # The learning schedule describes in which order which part of the network should be
            # trained and with which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'heatmaps_mse': ['hourglass'],
                        'radius_mse': ['radius'],
                    },
                    'learning_rate': 1e-3,
                },
            ],

            # Data sources for training (and testing).
            train_data={'synthetic': unityeyes},
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=100,
        )
