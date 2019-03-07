"""Concurrent testing during training."""
import collections
import platform
import threading
import time
import traceback

import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class LiveTester(object):
    """Manage concurrent testing on test data source."""

    def __init__(self, model, data_source, use_batch_statistics=True):
        """Initialize tester with reference to model and data sources."""
        self.model = model
        self.data = data_source
        self.time = self.model.time
        self.summary = self.model.summary
        self._tensorflow_session = model._tensorflow_session

        self._is_testing = False
        self._condition = threading.Condition()

        self._use_batch_statistics = use_batch_statistics

    def stop(self):
        logger.info('LiveTester::stop is being called.')
        self._is_testing = False

    def __del__(self):
        """Handle deletion of instance by closing thread."""
        if not hasattr(self, '_coordinator'):
            return
        self._coordinator.request_stop()
        with self._condition:
            self._is_testing = True  # Break wait if waiting
            self._condition.notify_all()
        self._coordinator.join([self._thread], stop_grace_period_secs=1)

    def _true_if_testing(self):
        return self._is_testing

    def trigger_test_if_not_testing(self, current_step):
        """If not currently testing, run test."""
        if not self._is_testing:
            with self._condition:
                self._is_testing = True
                self._testing_at_step = current_step
                self._condition.notify_all()

    def test_job(self):
        """Evaluate requested metric over entire test set."""
        while not self._coordinator.should_stop():
            with self._condition:
                self._condition.wait_for(self._true_if_testing)
                if self._coordinator.should_stop():
                    break
                should_stop = False
                try:
                    should_stop = self.do_full_test()
                except:
                    traceback.print_exc()
                self._is_testing = False
                if should_stop is True:
                    break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def do_full_test(self, sleep_between_batches=0.2):
        # Copy current weights over
        self.copy_model_weights()

        # Reset data sources
        for data_source_name, data_source in self.data.items():
            data_source.reset()
            num_batches = int(data_source.num_entries / data_source.batch_size)

        # Decide what to evaluate
        fetches = self._tensors_to_evaluate
        outputs = dict([(name, list()) for name in fetches.keys()])

        # Select random index to produce (image) summaries at
        summary_index = np.random.randint(num_batches)

        self.time.start('full test')
        for i in range(num_batches):
            if self._is_testing is not True:
                logger.debug('Testing flag found to be `False` at iter. %d' % i)
                break
            logger.debug('Testing on %03d/%03d batches.' % (i + 1, num_batches))
            if i == summary_index:
                fetches['summaries'] = self.summary.get_ops(mode='test')
            try:
                output = self._tensorflow_session.run(
                    fetches=fetches,
                    feed_dict={
                        self.model.is_training: False,
                        self.model.use_batch_statistics: self._use_batch_statistics,
                    },
                )
            except (tf.errors.CancelledError, RuntimeError):
                return True
            time.sleep(sleep_between_batches)  # Brief pause to prioritise training
            if 'summaries' in output:  # Write summaries on first batch
                self.summary.write_summaries(output['summaries'], self._testing_at_step)
                del fetches['summaries']
                del output['summaries']
            for name, value in output.items():  # Gather results from this batch
                outputs[name].append(output[name])
        self.time.end('full test')

        # If incomplete, skip this round of tests (most likely shutting down)
        if len(list(outputs.values())[0]) != num_batches:
            return True

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
        feed_dict = dict([(self._placeholders[name], value)
                         for name, value in outputs.items()])
        feed_dict[self.model.is_training] = False
        feed_dict[self.model.use_batch_statistics] = True
        try:
            summaries = self._tensorflow_session.run(
                fetches=self.summary.get_ops(mode='full_test'),
                feed_dict=feed_dict,
            )
        except (tf.errors.CancelledError, RuntimeError):
            return True
        self.summary.write_summaries(summaries, self._testing_at_step)

        return False

    def do_final_full_test(self, current_step):
        logger.info('Stopping the live testing threads.')

        # Stop thread(s)
        self._is_testing = False
        self._coordinator.request_stop()
        with self._condition:
            self._is_testing = True  # Break wait if waiting
            self._condition.notify_all()
        self._coordinator.join([self._thread], stop_grace_period_secs=1)

        # Start final full test
        logger.info('Running final full test')
        self.copy_model_weights()
        self._is_testing = True
        self._testing_at_step = current_step
        self.do_full_test(sleep_between_batches=0)

    def _post_model_build(self):
        """Prepare combined operation to copy model parameters over from CPU/GPU to CPU."""
        with tf.variable_scope('copy2test'):
            all_variables = tf.global_variables()
            train_vars = dict([(v.name, v) for v in all_variables
                               if not v.name.startswith('test/')])
            test_vars = dict([(v.name, v) for v in all_variables
                              if v.name.startswith('test/')])
            self._copy_variables_to_test_model_op = tf.tuple([
                test_vars['test/' + k].assign(train_vars[k]) for k in train_vars.keys()
                if 'test/' + k in test_vars
            ])

        # Begin testing thread
        self._coordinator = tf.train.Coordinator()
        self._thread = threading.Thread(target=self.test_job,
                                        name='%s_tester' % self.model.identifier)
        self._thread.daemon = True
        self._thread.start()

        # Pick tensors we need to evaluate
        all_tensors = dict(self.model.loss_terms['test'], **self.model.metrics['test'])
        self._tensors_to_evaluate = dict([(n, t) for n, t in all_tensors.items()])
        loss_terms_to_evaluate = dict([(n, t) for n, t in self.model.loss_terms['test'].items()
                                       if t in self._tensors_to_evaluate.values()])
        metrics_to_evaluate = dict([(n, t) for n, t in self.model.metrics['test'].items()
                                    if t in self._tensors_to_evaluate.values()])

        # Placeholders for writing summaries at end of test run
        self._placeholders = {}
        for type_, tensors in (('loss', loss_terms_to_evaluate),
                               ('metric', metrics_to_evaluate)):
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
