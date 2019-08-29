"""Base model class for Tensorflow-based model construction."""
from .data_source import BaseDataSource
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from .live_tester import LiveTester
from .time_manager import TimeManager
from .summary_manager import SummaryManager
from .checkpoint_manager import CheckpointManager
import logging
logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base model class for Tensorflow-based model construction.

    This class assumes that there exist no other Tensorflow models defined.
    That is, any variable that exists in the Python session will be grabbed by the class.
    """

    def __init__(self,
                 tensorflow_session: tf.Session,
                 learning_schedule: List[Dict[str, Any]] = [],
                 train_data: Dict[str, BaseDataSource] = {},
                 test_data: Dict[str, BaseDataSource] = {},
                 test_losses_or_metrics: str = None,
                 use_batch_statistics_at_test: bool = True,
                 identifier: str = None):
        """Initialize model with data sources and parameters."""
        self._tensorflow_session = tensorflow_session
        self._train_data = train_data
        self._test_data = test_data
        self._test_losses_or_metrics = test_losses_or_metrics
        self._initialized = False
        self.__identifier = identifier

        # Extract and keep known prefixes/scopes
        self._learning_schedule = learning_schedule
        self._known_prefixes = [schedule for schedule in learning_schedule]

        # Check consistency of given data sources
        train_data_sources = list(train_data.values())
        test_data_sources = list(test_data.values())
        all_data_sources = train_data_sources + test_data_sources
        first_data_source = all_data_sources.pop()
        self._batch_size = first_data_source.batch_size
        self._data_format = first_data_source.data_format
        for data_source in all_data_sources:
            if data_source.batch_size != self._batch_size:
                raise ValueError(('Data source "%s" has anomalous batch size of %d ' +
                                  'when detected batch size is %d.') % (data_source.short_name,
                                                                        data_source.batch_size,
                                                                        self._batch_size))
            if data_source.data_format != self._data_format:
                raise ValueError(('Data source "%s" has anomalous data_format of %s ' +
                                  'when detected data_format is %s.') % (data_source.short_name,
                                                                         data_source.data_format,
                                                                         self._data_format))
        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        # Log messages to file
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self.output_path + '/messages.log')
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        for handler in root_logger.handlers[1:]:  # all except stdout
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)

        # Register a manager for tf.Summary
        self.summary = SummaryManager(self)

        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)

        # Register a manager for timing related operations
        self.time = TimeManager(self)

        # Prepare for live (concurrent) validation/testing during training, on the CPU
        self._enable_live_testing = (len(self._train_data) > 0) and (len(self._test_data) > 0)
        self._tester = LiveTester(self, self._test_data, use_batch_statistics_at_test)

        # Run-time parameters
        with tf.variable_scope('learning_params'):
            self.is_training = tf.placeholder(tf.bool)
            self.use_batch_statistics = tf.placeholder(tf.bool)
            self.learning_rate_multiplier = tf.Variable(1.0, trainable=False, dtype=tf.float32)
            self.learning_rate_multiplier_placeholder = tf.placeholder(dtype=tf.float32)
            self.assign_learning_rate_multiplier = \
                tf.assign(self.learning_rate_multiplier, self.learning_rate_multiplier_placeholder)

        self._build_all_models()

    def __del__(self):
        """Explicitly call methods to cleanup any live threads."""
        train_data_sources = list(self._train_data.values())
        test_data_sources = list(self._test_data.values())
        all_data_sources = train_data_sources + test_data_sources
        for data_source in all_data_sources:
            data_source.cleanup()
        self._tester.__del__()

    __identifier_stem = None

    @property
    def identifier(self):
        """Identifier for model based on time."""
        if self.__identifier is not None:  # If loading from checkpoints or having naming enforced
            return self.__identifier
        if self.__identifier_stem is None:
            self.__identifier_stem = self.__class__.__name__ + '/' + time.strftime('%y%m%d%H%M%S')
        return self.__identifier_stem + self._identifier_suffix

    @property
    def _identifier_suffix(self):
        """Identifier suffix for model based on data sources and parameters."""
        return ''

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + '/../../outputs'),
                          self.identifier)

    def _build_all_models(self):
        """Build training (GPU/CPU) and testing (CPU) streams."""
        self.output_tensors = {}
        self.loss_terms = {}
        self.metrics = {}

        def _build_datasource_summaries(data_sources, mode):
            """Register summary operations for input data from given data sources."""
            with tf.variable_scope('%s_data' % mode):
                for data_source_name, data_source in data_sources.items():
                    tensors = data_source.output_tensors
                    for key, tensor in tensors.items():
                        summary_name = '%s/%s' % (data_source_name, key)
                        shape = tensor.shape.as_list()
                        num_dims = len(shape)
                        if num_dims == 4:  # Image data
                            if shape[1] == 1 or shape[1] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_first')
                            elif shape[3] == 1 or shape[3] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_last')
                            # TODO: fix issue with no summary otherwise
                        elif num_dims == 2:
                            self.summary.histogram(summary_name, tensor)
                        else:
                            logger.debug('I do not know how to create a summary for %s (%s)' %
                                         (summary_name, tensor.shape.as_list()))

        def _build_train_or_test(mode):
            data_sources = self._train_data if mode == 'train' else self._test_data

            # Build model
            output_tensors, loss_terms, metrics = self.build_model(data_sources, mode=mode)

            # Record important tensors
            self.output_tensors[mode] = output_tensors
            self.loss_terms[mode] = loss_terms
            self.metrics[mode] = metrics

            # Create summaries for scalars
            if mode == 'train':
                for name, loss_term in loss_terms.items():
                    self.summary.scalar('loss/%s/%s' % (mode, name), loss_term)
                for name, metric in metrics.items():
                    self.summary.scalar('metric/%s/%s' % (mode, name), metric)

        # Build the main model
        if len(self._train_data) > 0:
            _build_datasource_summaries(self._train_data, mode='train')
            _build_train_or_test(mode='train')
            logger.info('Built model.')

            # Print no. of parameters and lops
            flops = tf.profiler.profile(
                options=tf.profiler.ProfileOptionBuilder(
                    tf.profiler.ProfileOptionBuilder.float_operation()
                ).with_empty_output().build())
            logger.info('------------------------------')
            logger.info(' Approximate Model Statistics ')
            logger.info('------------------------------')
            logger.info('FLOPS per input: {:,}'.format(flops.total_float_ops / self._batch_size))
            logger.info(
                'Trainable Parameters: {:,}'.format(
                    np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
                )
            )
            logger.info('------------------------------')

        # If there are any test data streams, build same model with different scope
        # Trainable parameters will be copied at test time
        if len(self._test_data) > 0:
            _build_datasource_summaries(self._test_data, mode='test')
            with tf.variable_scope('test'):
                _build_train_or_test(mode='test')
            logger.info('Built model for live testing.')

        if self._enable_live_testing:
            self._tester._post_model_build()  # Create copy ops to be run before every test run

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def initialize_if_not(self, training=False):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Build supporting operations
        with tf.variable_scope('savers'):
            self.checkpoint.build_savers()  # Create savers
        if training:
            with tf.variable_scope('optimize'):
                self._build_optimizers()

        # Start pre-processing routines
        for _, datasource in self._train_data.items():
            datasource.create_and_start_threads()

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    def _build_optimizers(self):
        """Based on learning schedule, create optimizer instances."""
        self._optimize_ops = []
        all_trainable_variables = tf.trainable_variables()
        all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        all_reg_losses = tf.losses.get_regularization_losses()
        for spec in self._learning_schedule:
            optimize_ops = []
            update_ops = []
            loss_terms = spec['loss_terms_to_optimize']
            reg_losses = []
            assert isinstance(loss_terms, dict)
            for loss_term_key, prefixes in loss_terms.items():
                assert loss_term_key in self.loss_terms['train'].keys()
                variables_to_train = []
                for prefix in prefixes:
                    variables_to_train += [
                        v for v in all_trainable_variables
                        if v.name.startswith(prefix)
                    ]
                    update_ops += [
                        o for o in all_update_ops
                        if o.name.startswith(prefix)
                    ]
                    reg_losses += [
                        l for l in all_reg_losses
                        if l.name.startswith(prefix)
                    ]

                optimizer_class = tf.train.AdamOptimizer
                optimizer = optimizer_class(
                    learning_rate=self.learning_rate_multiplier * spec['learning_rate'],
                    # beta1=0.9,
                    # beta2=0.999,
                )
                final_loss = self.loss_terms['train'][loss_term_key]
                if len(reg_losses) > 0:
                    final_loss += tf.reduce_sum(reg_losses)
                with tf.control_dependencies(update_ops):
                    gradients, variables = zip(*optimizer.compute_gradients(
                        loss=final_loss,
                        var_list=variables_to_train,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
                    ))
                    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)  # TODO: generalize
                    optimize_op = optimizer.apply_gradients(zip(gradients, variables))
                optimize_ops.append(optimize_op)
            self._optimize_ops.append(optimize_ops)
            logger.info('Built optimizer for: %s' % ', '.join(loss_terms.keys()))

    def train_loop_pre(self, current_step):
        """Run this at beginning of training loop."""
        pass

    def train_loop_post(self, current_step):
        """Run this at end of training loop."""
        pass

    def train(self, num_epochs=None, num_steps=None):
        """Train model as requested."""
        if num_steps is None:
            num_entries = np.min([s.num_entries for s in list(self._train_data.values())])
            num_steps = int(num_epochs * num_entries / self._batch_size)
        self.initialize_if_not(training=True)

        try:
            initial_step = self.checkpoint.load_all()
            current_step = initial_step
            for current_step in range(initial_step, num_steps):
                # Extra operations defined in implementation of this base class
                self.train_loop_pre(current_step)

                # Select loss terms, optimize operations, and metrics tensors to evaluate
                fetches = {}
                schedule_id = current_step % len(self._learning_schedule)
                schedule = self._learning_schedule[schedule_id]
                fetches['optimize_ops'] = self._optimize_ops[schedule_id]
                loss_term_keys, _ = zip(*list(schedule['loss_terms_to_optimize'].items()))
                fetches['loss_terms'] = [self.loss_terms['train'][k] for k in loss_term_keys]
                summary_op = self.summary.get_ops(mode='train')
                if len(summary_op) > 0:
                    fetches['summaries'] = summary_op

                # Run one optimization iteration and retrieve calculated loss values
                self.time.start('train_iteration', average_over_last_n_timings=100)
                outcome = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict={
                        self.is_training: True,
                        self.use_batch_statistics: True,
                    }
                )
                self.time.end('train_iteration')

                # Print progress
                to_print = '%07d> ' % current_step
                to_print += ', '.join(['%s = %g' % (k, v)
                                       for k, v in zip(loss_term_keys, outcome['loss_terms'])])
                self.time.log_every('train_iteration', to_print, seconds=2)

                # Trigger copy weights & concurrent testing (if not already running)
                if self._enable_live_testing:
                    self._tester.trigger_test_if_not_testing(current_step)

                # Write summaries
                if 'summaries' in outcome:
                    self.summary.write_summaries(outcome['summaries'], current_step)

                # Save model weights
                if self.time.has_been_n_seconds_since_last('save_weights', 600) \
                        and current_step > initial_step:
                    self.checkpoint.save_all(current_step)

                # Extra operations defined in implementation of this base class
                self.train_loop_post(current_step)

        except KeyboardInterrupt:
            # Handle CTRL-C graciously
            self.checkpoint.save_all(current_step)
            sys.exit(0)

        # Stop live testing, and run final full test
        if self._enable_live_testing:
            self._tester.do_final_full_test(current_step)

        # Save final weights
        if current_step > initial_step:
            self.checkpoint.save_all(current_step)

    def inference_generator(self):
        """Perform inference on test data and yield a batch of output."""
        self.initialize_if_not(training=False)
        self.checkpoint.load_all()  # Load available weights

        # TODO: Make more generic by not picking first source
        data_source = next(iter(self._train_data.values()))
        while True:
            fetches = dict(self.output_tensors['train'], **data_source.output_tensors)
            start_time = time.time()
            outputs = self._tensorflow_session.run(
                fetches=fetches,
                feed_dict={
                    self.is_training: False,
                    self.use_batch_statistics: True,
                },
            )
            outputs['inference_time'] = 1e3*(time.time() - start_time)
            yield outputs
