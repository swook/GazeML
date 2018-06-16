"""Concurrent testing during training."""
import threading
import time

import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class LiveTester(object):
    """Manage concurrent testing on test data source."""

    def __init__(self, model, data_source):
        """Initialize tester with reference to model and data sources."""
        self.model = model
        self.data = data_source
        self.time = self.model.time
        self.summary = self.model.summary
        self._tensorflow_session = model._tensorflow_session

        self._is_testing = False
        self._condition = threading.Condition()

    def __del__(self):
        """Handle deletion of instance by closing thread."""
        pass

    def _true_if_testing(self):
        return self._is_testing

    def trigger_test_if_not_testing(self, current_step):
        """If not currently testing, run test."""
        if not self._is_testing:
            with self._condition:
                self.copy_model_weights()
                self._is_testing = True
                self._testing_at_step = current_step
                self._condition.notify_all()

    def test_job(self):
        """Evaluate requested metric over entire test set."""
        while not self._coordinator.should_stop():
            with self._condition:
                self._condition.wait_for(self._true_if_testing)

                # Reset data sources
                for data_source_name, data_source in self.data.items():
                    data_source.reset()
                    num_batches = int(data_source.num_entries / data_source.batch_size)

                # Decide what to evaluate
                fetches = dict([
                    (name, self._tensors_to_evaluate[name])
                    for name in self.model._test_losses_or_metrics
                ])
                outputs = dict([(name, list()) for name in fetches.keys()])

                # Get summary ops but remove full_test ones
                fetches['summaries'] = self.summary.get_ops(mode='test')

                self.time.start('full test')
                for i in range(num_batches):
                    # logger.debug('Tested on %03d/%03d batches.' % (i + 1, num_batches))
                    try:
                        output = self._tensorflow_session.run(
                            fetches=fetches,
                            feed_dict={
                                self.model.is_training: False,
                                self.model.use_batch_statistics: True,
                            },
                        )
                    except (tf.errors.CancelledError, RuntimeError):
                        return
                    time.sleep(0.1)  # Brief pause to prioritise training
                    if 'summaries' in output:  # Write summaries on first batch
                        self.summary.write_summaries(output['summaries'], self._testing_at_step)
                        del fetches['summaries']
                        del output['summaries']
                    for name, value in output.items():  # Gather results from this batch
                        outputs[name].append(output[name])
                self.time.end('full test')

                # Calculate mean values
                for name, values in outputs.items():
                    outputs[name] = np.mean(values)

                # TODO: Log metric as summary
                to_print = '[Test at step %06d] ' % self._testing_at_step
                to_print += ', '.join([
                    '%s = %f' % (name, value) for name, value in outputs.items()
                ])
                logger.info(to_print)

                # Store mean metrics/losses (and other summaries)
                feed_dict = dict([(self._placeholders[name], outputs[name])
                                 for name in self._tensors_to_evaluate.keys()])
                feed_dict[self.model.is_training] = False
                feed_dict[self.model.use_batch_statistics] = True
                try:
                    summaries = self._tensorflow_session.run(
                        fetches=self.summary.get_ops(mode='full_test'),
                        feed_dict=feed_dict,
                    )
                except (tf.errors.CancelledError, RuntimeError):
                    return
                self.summary.write_summaries(summaries, self._testing_at_step)

                self._is_testing = False

    def _post_model_build(self):
        """Prepare combined operation to copy model parameters over from CPU/GPU to CPU."""
        with tf.variable_scope('copy2test'):
            all_trainable_variables = tf.trainable_variables()
            train_vars = dict([(v.name, v) for v in all_trainable_variables
                               if not v.name.startswith('test/')])
            test_vars = dict([(v.name, v) for v in all_trainable_variables
                              if v.name.startswith('test/')])
            assert len(train_vars) == len(test_vars)
            self._copy_variables_to_test_model_op = tf.tuple([
                test_vars['test/' + k].assign(train_vars[k]) for k in train_vars.keys()
            ])

        # Begin testing thread
        self._coordinator = tf.train.Coordinator()
        self._thread = threading.Thread(target=self.test_job,
                                        name='%s_tester' % self.model.identifier)
        self._thread.daemon = True
        self._thread.start()

        # Pick tensors we need to evaluate
        self._tensors_to_evaluate = dict(self.model.loss_terms['test'],
                                         **self.model.metrics['test'])

        # Placeholders for writing summaries at end of test run
        self._placeholders = {}
        for type_, tensors in (('loss', self.model.loss_terms['test']),
                               ('metric', self.model.metrics['test'])):
            for name in tensors.keys():
                name = '%s/test/%s' % (type_, name)
                placeholder = tf.placeholder(dtype=np.float32, name=name + '_placeholder')
                self.summary.scalar(name, placeholder)
                self._placeholders[name.split('/')[-1]] = placeholder

    def copy_model_weights(self):
        """Copy weights from main model used for training.

        This operation should stop-the-world, that is, training should not occur.
        """
        assert self._copy_variables_to_test_model_op is not None
        self._tensorflow_session.run(self._copy_variables_to_test_model_op)
        logger.debug('Copied over trainable model parameters for testing.')
