"""Manage saving and loading of model checkpoints."""
import os
import re

import tensorflow as tf


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
        all_prefixes = []
        for schedule in self._model._learning_schedule:
            for prefixes in schedule['loss_terms_to_optimize'].values():
                all_prefixes += prefixes
        all_prefixes = list(set(all_prefixes))

        # For each prefix, create saver
        self._savers = {}
        for prefix in all_prefixes:
            vars_to_save = [v for v in all_saveable_vars if v.name.startswith(prefix + '/')]
            if len(vars_to_save):
                self._savers[prefix] = tf.train.Saver(vars_to_save, max_to_keep=2)

    def load_all(self):
        """Load all available weights for each known prefix."""
        iteration_number = 0
        for prefix, saver in self._savers.items():
            output_path = '%s/checkpoints/%s' % (self._model.output_path, prefix)
            checkpoint = tf.train.get_checkpoint_state(output_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
                try:  # Attempt to restore saveable variables
                    self._savers[prefix].restore(self._tensorflow_session,
                                                 '%s/%s' % (output_path, checkpoint_name))
                    iteration_number = \
                        int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        return iteration_number

    def save_all(self, iteration_number):
        """Save all prefixes."""
        for prefix, saver in self._savers.items():
            output_path = '%s/checkpoints/%s' % (self._model.output_path, prefix)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            saver.save(self._tensorflow_session, output_path + '/model',
                       global_step=iteration_number)
