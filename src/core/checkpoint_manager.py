"""Manage saving and loading of model checkpoints."""
import os
import re

import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class CheckpointManager(object):
    """Manager to coordinate saving and loading of trainable parameters."""

    def __init__(self, model):
        """Initialize manager based on given model instance."""
        self._tensorflow_session = model._tensorflow_session
        self._model = model

    def build_savers(self):
        """Create tf.train.Saver instances."""
        all_saveable_vars = sorted(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
                                   tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS) +
                                   tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES) +
                                   tf.get_collection_ref('batch_norm_non_trainable'),
                                   key=lambda v: v.name)

        # Grab all available prefixes
        all_prefixes = []
        for v in all_saveable_vars:
            name = v.name
            if '/' not in name:
                continue
            prefix = name.split('/')[0]
            if prefix == 'test' or prefix == 'learning_params':
                continue
            if prefix not in all_prefixes:
                all_prefixes.append(prefix)

        # For each prefix, create saver
        self._savers = {}
        for prefix in all_prefixes:
            vars_to_save = [v for v in all_saveable_vars if v.name.startswith(prefix + '/')]
            if len(vars_to_save):
                self._savers[prefix] = tf.train.Saver(vars_to_save, max_to_keep=2)

    def load_all(self):
        """Load all available weights for each known prefix."""
        iteration_number = 0
        iteration_numbers = []
        for prefix, saver in self._savers.items():
            output_path = '%s/checkpoints/%s' % (self._model.output_path, prefix)
            checkpoint = tf.train.get_checkpoint_state(output_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
                try:  # Attempt to restore saveable variables
                    self._savers[prefix].restore(self._tensorflow_session,
                                                 '%s/%s' % (output_path, checkpoint_name))
                    iteration_numbers.append(
                        int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        if len(iteration_numbers) > 0:
            iteration_number = np.amax(iteration_numbers)
        return iteration_number

    def save_all(self, iteration_number):
        """Save all prefixes."""
        prefixes_to_use = []
        for schedule in self._model._learning_schedule:
            for prefixes in schedule['loss_terms_to_optimize'].values():
                prefixes_to_use += prefixes
        prefixes_to_use = list(set(prefixes_to_use))

        for prefix, saver in self._savers.items():
            if prefix not in prefixes_to_use:
                continue
            output_path = '%s/checkpoints/%s' % (self._model.output_path, prefix)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            saver.save(self._tensorflow_session, output_path + '/model',
                       global_step=iteration_number)
            logger.debug('Saved %s' % output_path)
        logger.info('CheckpointManager::save_all call done')
